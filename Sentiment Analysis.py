from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
def analyze_sentiment(text: str):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probabilities).item()
    confidence = probabilities[0][label_id].item()
    label = "POSITIVE" if label_id == 1 else "NEGATIVE"
    return label, confidence
if __name__ == "__main__":
    text = input("Enter text: ")
    label, confidence = analyze_sentiment(text)
    print(f"Sentiment: {label} (confidence: {confidence:.2f})")
