import torch
import matplotlib.pyplot as plt
import joblib
from model import ReviewModel
from data import preprocess_text


def evaluate(model, test_loader):
    
    model.eval()
    correct=0
    total=0
    
    with torch.no_grad():
        for X, y in test_loader:
            outputs=model(X)
            preds=outputs.argmax(dim=1)
            correct+=(preds==y).sum().item()
            total+=y.size(0)
    
    return correct / total


def plot_losses(train_losses, val_losses, save_path='training_curves.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()


def predict_sentiment(review_text, model_path='sentiment_model.pkl', vocab_path='vocab.pkl'):
    vocab=joblib.load(vocab_path)
    state_dict=joblib.load(model_path)
    
    model = ReviewModel(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64, num_classes=2)
    model.load_state_dict(state_dict)
    model.eval()
    
    words = preprocess_text(review_text)
    max_len = 128
    unk_idx = vocab['<UNK>']
    tokens = [vocab.get(word, unk_idx) for word in words]
    
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    with torch.no_grad():
        X = torch.LongTensor([tokens])
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    label = 'positive' if pred_class == 1 else 'negative'
    return label, confidence