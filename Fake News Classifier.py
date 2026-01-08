from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def detect_fake_news(text: str):
    """
    Detect whether a news headline is fake or real.

    Returns:
        label (str): FAKE or REAL
        confidence (float): model confidence score
    """
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

    label = "FAKE" if label_id == 1 else "REAL"
    return label, confidence

if __name__ == "__main__":
    text = input("Enter news text: ")
    label, confidence = detect_fake_news(text)
    print(f"News: {label} (confidence: {confidence:.2f})")
