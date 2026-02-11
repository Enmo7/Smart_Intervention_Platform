import torch
import joblib # Using joblib for better cross-platform compatibility
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Configuration - Paths are relative to the project root
MODEL_NAME = "aubmindlab/bert-base-arabertv02" 
WEIGHTS_PATH = "./model/best_model.pth"
ENCODER_PATH = "./model/label_encoder.pkl"

# Initialize Global Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_encoder():
    """
    Load the LabelEncoder using joblib.
    Joblib is more robust than pickle for scikit-learn objects across Windows/Linux.
    """
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at: {ENCODER_PATH}")
    
    try:
        return joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"Error loading encoder with joblib: {e}")
        raise

# Load encoder and determine number of classes
label_encoder = load_encoder()
NUM_LABELS = len(label_encoder.classes_)

def load_custom_model():
    """
    Initialize the BERT model architecture and load weights from the .pth file.
    """
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_config(config)
    
    if os.path.exists(WEIGHTS_PATH):
        # Map location to CPU for systems without a dedicated GPU
        state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode
        print(f"AI Engine: Model weights loaded successfully ({NUM_LABELS} classes).")
    return model

model = load_custom_model()

def preprocess_arabic(text):
    """Clean and normalize Arabic text: remove URLs, handles, and standardize characters."""
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    return text.strip()

def analyze_message(message):
    """
    Main inference pipeline: Preprocess -> Tokenize -> Predict -> Map to Intervention.
    """
    cleaned_text = preprocess_arabic(message)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_idx = torch.max(probs, dim=-1)
    
    # Map the predicted ID back to the string label (e.g., High, Medium, Low)
    label_name = label_encoder.inverse_transform([predicted_idx.item()])[0]
    
    # Import logic module locally to avoid circular dependencies
    from app.logic import get_intervention
    intervention = get_intervention(label_name)
    
    return {
        "label": label_name,
        "confidence": round(confidence.item() * 100, 2),
        **intervention
    }