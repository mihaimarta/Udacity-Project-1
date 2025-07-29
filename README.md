# Lightweight Fine-Tuning for Emotion Classification

This project demonstrates lightweight fine-tuning of a Hugging Face transformer model for emotion classification using the `dair-ai/emotion` dataset. It includes parameter-efficient fine-tuning (PEFT) with LoRA and compares the performance of the base and fine-tuned models.

## Features
- Loads and preprocesses the `dair-ai/emotion` dataset
- Fine-tunes `distilbert-base-uncased` for emotion classification
- Implements PEFT using LoRA for efficient adaptation
- Evaluates and compares base and LoRA model accuracy
- Includes code for inference on custom samples

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Open `LightweightFineTuning.ipynb` in Jupyter or VS Code.
2. Run the notebook cells sequentially:
   - Load and inspect the dataset
   - Tokenize and preprocess data
   - Train and evaluate the base model
   - Apply LoRA and train the PEFT model
   - Compare accuracy and run inference

## Project Structure
- `LightweightFineTuning.ipynb`: Main notebook with all code and experiments
- `requirements.txt`: Python dependencies
- `README.md`: Project overview and instructions

## Tips for Better Accuracy
- Use more training data if possible
- Tune LoRA hyperparameters (`r`, `lora_alpha`, `lora_dropout`)
- Train for more epochs
- Adjust learning rate
- Unfreeze more model layers for adaptation

## References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)

## License
This project is for educational purposes.
