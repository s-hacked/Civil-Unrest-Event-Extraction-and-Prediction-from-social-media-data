import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/xlm_roberta_bilingual_event_detector"

print("Loading bilingual XLM-R model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    prediction = torch.argmax(probabilities).item()
    event_probability = probabilities[1].item()

    print("\n" + "=" * 50)
    print("INPUT:", text)
    print("PREDICTION:", "EVENT" if prediction == 1 else "NON-EVENT")
    print("EVENT PROBABILITY:", round(event_probability * 100, 2), "%")
    print("=" * 50)


# English
predict("Thousands of students protested in Delhi yesterday.")

# Hindi
predict("दिल्ली में हजारों छात्रों ने विरोध प्रदर्शन किया।")