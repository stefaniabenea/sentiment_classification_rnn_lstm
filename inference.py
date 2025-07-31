import torch
from utils import tokenizer  
from models import RNNClassifier, LSTMClassifier
import argparse

def predict(text, model, device):
    model.eval()
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs[0, pred_class].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for sentiment classification")
    parser.add_argument("--model", choices=['rnn', 'lstm'], required=True, help="Model type")
    
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'rnn':
        model = RNNClassifier(vocab_size=tokenizer.vocab_size, embed_size=100, hidden_dim=64, num_classes=2, num_layers = 1)
    elif args.model == 'lstm':
        model = LSTMClassifier(vocab_size=tokenizer.vocab_size, embed_size=100, hidden_dim=64, num_classes=2, num_layers=1)

    model.load_state_dict(torch.load(f"models/{args.model}.pth", map_location=device))
    model.to(device)

    pred_class, confidence = predict(args.text, model, device)
    label_map = {0: "Negative", 1: "Positive"}
    print(f"Prediction: {label_map[pred_class]} (confidence: {confidence:.2f})")
