import os
import pandas as pd
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.utils.data import Dataset, DataLoader, random_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from torch.optim import AdamW
from transformers import get_scheduler
import evaluate
from pyxlsb import open_workbook
from torch import amp  # Updated import for mixed precision

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset class
class CustomAudioDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.2),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.2),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
            Shift(min_shift=-0.2, max_shift=0.2, p=0.2),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['Video'].strip()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        transcript = self.df.iloc[idx]['Transcripts']

        # Check if transcript is valid
        if not isinstance(transcript, str) or transcript.strip() == "":
            print(f"Invalid transcript at index {idx}. Skipping.")
            return None  # Handle as needed

        # Load and preprocess the audio
        audio, sample_rate = torchaudio.load(audio_path)
        audio = self.augment(samples=audio.numpy().squeeze(), sample_rate=sample_rate)

        # Process the audio to extract features using the processor
        inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features[0]

        # Tokenize the transcript correctly
        try:
            labels = self.processor.tokenizer(text_target=transcript, return_tensors="pt").input_ids[0]
        except ValueError as e:
            print(f"Error tokenizing transcript: {transcript}. Error: {e}")
            labels = torch.tensor([])  # Handle empty labels gracefully

        return {
            "input_features": input_features,
            "labels": labels,
            "transcript": transcript
        }

# Custom Data Collator
class DataCollatorWithPadding:
    def __init__(self, processor, max_length=448):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        input_features = [example["input_features"] for example in batch if example is not None]
        labels = [example["labels"][:self.max_length] for example in batch if example is not None]  # Truncate labels

        batch_inputs = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        batch_labels = self.processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")

        batch_labels["input_ids"] = batch_labels["input_ids"].masked_fill(batch_labels["attention_mask"].ne(1), -100)

        return {
            "input_features": batch_inputs["input_features"].to(device),
            "labels": batch_labels["input_ids"].to(device)
        }

# Load your transcript data
def load_xlsb_data(xlsb_file):
    data = []
    with open_workbook(xlsb_file) as wb:
        with wb.get_sheet(1) as sheet:
            for row in sheet.rows():
                data.append([item.v for item in row])
    df = pd.DataFrame(data[1:], columns=data[0])  # Assuming first row is header
    return df

# Function to save model checkpoints
def save_checkpoint(model, processor, optimizer, scheduler, epoch, save_dir, checkpoint_name):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(os.path.join(save_dir, checkpoint_name))
    processor.save_pretrained(os.path.join(save_dir, checkpoint_name))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(save_dir, checkpoint_name, 'checkpoint.pth'))
    print(f"Checkpoint saved at epoch {epoch}.")

# Custom training loop with validation and early stopping
def train(model, dataloader, val_dataloader, optimizer, scheduler, processor, num_epochs=10, save_interval=1, save_dir="checkpoints", patience=2):
    model.train()
    wer_metric = evaluate.load("wer")  # Load WER metric
    scaler = amp.GradScaler()  # For mixed precision training

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:  # Skip any None entries
                continue
            try:
                with amp.autocast():  # Mixed precision
                    outputs = model(input_features=batch["input_features"], labels=batch["labels"])
                    loss = outputs.loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()  # Scale loss for mixed precision
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()

                # Print batch progress
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item()}")

                # Evaluate WER every 20 batches (more frequent due to small dataset)
                if (batch_idx + 1) % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        predictions = model.generate(input_features=batch["input_features"])
                        pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
                        labels = batch["labels"].cpu().numpy()
                        label_str = processor.batch_decode(labels, skip_special_tokens=True)

                        # Compute WER
                        wer = wer_metric.compute(predictions=pred_str, references=label_str)
                        print(f"Batch {batch_idx + 1}, WER: {wer:.4f}")

                        # Log predictions and actual transcripts if WER is high
                        if wer >= 1.0:  # Adjust threshold as necessary
                            for i in range(len(pred_str)):
                                print(f"Predicted: {pred_str[i]} | Actual: {label_str[i]}")

                    model.train()

            except Exception as e:
                print(f"Error in Batch {batch_idx + 1}: {e}")
                continue

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

        # Validation step
        val_loss = validate(model, val_dataloader, processor)
        print(f"Validation Loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, processor, optimizer, scheduler, epoch + 1, save_dir, "checkpoint_best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# Validation function
def validate(model, val_dataloader, processor):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            if batch is None:  # Skip any None entries
                continue
            outputs = model(input_features=batch["input_features"], labels=batch["labels"])
            total_loss += outputs.loss.item()
    return total_loss / len(val_dataloader)

# Main code block
if __name__ == "__main__":
    XLSB_FILE_PATH = r'C:\MAJOR PROJ\fine tune whisper\Nepali Speech To Text Dataset\transcripts\audio transcript.xlsb'

    # Load data from the .xlsb file
    transcript_data = load_xlsb_data(XLSB_FILE_PATH)
    print(f"Loaded {len(transcript_data)} samples from the dataset.")

    # Use Whisper Large Model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)  # Move model to GPU
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="Nepali", task="transcribe")

    # Create the custom dataset
    dataset = CustomAudioDataset(transcript_data, processor)

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    data_collator = DataCollatorWithPadding(processor, max_length=448)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)

    # Set up training
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * 10)

    # Train the model and compute WER
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, processor, num_epochs=5)

    # Save the final trained model
    model.save_pretrained(r'C:\MAJOR PROJ\fine tune whisper\trained-whisper-model-large')
    processor.save_pretrained(r'C:\MAJOR PROJ\fine tune whisper\trained-whisper-model-large')
    print("Model saved successfully.")
