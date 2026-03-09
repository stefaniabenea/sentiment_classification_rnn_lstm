import torch
import inference
import pytest
from models import LSTMClassifier, RNNClassifier


@pytest.mark.parametrize("model_class", [RNNClassifier, LSTMClassifier])
@pytest.mark.parametrize("seq_len",[1,3,10])
def test_predict_with_real_model_and_fake_tokenizer(monkeypatch, model_class, seq_len):
    device = torch.device("cpu")

    def fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": torch.randint(0,50, (1,seq_len),dtype=torch.long)}

    monkeypatch.setattr(inference, "tokenizer", fake_tokenizer)

    vocab_size = 100
    embed_size = 16
    hidden_dim = 32
    num_classes = 2
    num_layers = 1

    model = model_class(
        vocab_size,
        embed_size,
        hidden_dim,
        num_classes,
        num_layers
    )
    model.eval()

    pred_class, confidence = inference.predict("this is a test", model, device)

    assert pred_class in [0, 1]
    assert isinstance(pred_class, int)
    assert 0.0 <= confidence <= 1.0
    assert isinstance(confidence, float)