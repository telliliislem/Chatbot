import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to save the pre-trained model
model_path = "models/fine_tuned_model"

# Ensure the directory exists
os.makedirs(model_path, exist_ok=True)

# Specify the model to download
model_name = "xlm-roberta-base"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)

# Save the tokenizer and model to the specified path
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print(f"Model and tokenizer saved to {model_path}")
