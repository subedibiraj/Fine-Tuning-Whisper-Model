# Whisper AI Model Training and Evaluation (Fine Tuning)

This repository contains a script to train and evaluate a Whisper AI model using the Hugging Face Transformers library.

## Prerequisites

Ensure you have the following installed on your machine:
- Python 3.8 or later
- CUDA 12.4
- cuDNN 8.9.4
- Git

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/subedibiraj/Fine-Tuning-Whisper-Model
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the necessary Python packages:**
    ```bash
    pip install torch==2.0.1+cu124 torchvision==0.15.2+cu124 torchaudio==2.0.2+cu124 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```

## Usage

1. **Log in to Hugging Face:**
    Get your Hugging Face token from [here](https://huggingface.co/settings/tokens) and replace the placeholder in the script.

2. **Run the script:**
    ```bash
    python script.py
    ```

3. **Monitor the training:**
    Training progress and logs will be saved in the `./whisper-small-ne-NP` directory and can be viewed using TensorBoard:
    ```bash
    tensorboard --logdir ./whisper-small-ne-NP
    ```

## Notes

- Ensure that you have CUDA and cuDNN correctly installed and configured. 
- The training process can take several hours depending on your GPU.

## Files

- `script.py`: The main script to train and evaluate the Whisper AI model.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the Transformers library and datasets.
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for the dataset.
