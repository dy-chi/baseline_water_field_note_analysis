from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Assuming "models" is the directory where you saved your model
best_model_path = "models"  # Change this path accordingly

# Load the best model
best_model = BertForSequenceClassification.from_pretrained(best_model_path)

# Optionally, load the corresponding tokenizer if needed
tokenizer = BertTokenizer.from_pretrained(best_model_path)

# Move the model to the appropriate device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model.to(device)

# Now, you can use the loaded best_model for inference or further training.
