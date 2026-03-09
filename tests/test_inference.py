import torch
import inference
import pytest

class DummyModel(torch.nn.Module):
  
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, input_ids):
        return self._logits
    
def test_predict_returns_class_and_confidence(monkeypatch):
    device = torch.device("cpu")
    
    def fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}
    
    monkeypatch.setattr(inference,"tokenizer", fake_tokenizer)
    model = DummyModel(torch.tensor([[1.0,3.0]], dtype=torch.float32))
    pred_class, confidence = inference.predict("any text", model, device)

    assert pred_class == 1
    assert 0.0 <= confidence <= 1.0

def test_predict_picks_class_0(monkeypatch):
    device = torch.device("cpu")

    def fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[9, 9]], dtype=torch.long)}

    monkeypatch.setattr(inference, "tokenizer", fake_tokenizer)

    
    model = DummyModel(torch.tensor([[5.0, 1.0]], dtype=torch.float32))

    pred_class, confidence = inference.predict("x", model, device)

    assert pred_class == 0
    assert 0.0 <= confidence <= 1.0

def test_predict_with_mocker(mocker):

    device = torch.device("cpu")

    def fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1,2,3,4]], dtype=torch.long)}

    mocker.patch.object(inference, "tokenizer", side_effect=fake_tokenizer)

    model = DummyModel(torch.tensor([[1.0,3.0]]))

    pred_class, confidence = inference.predict("text", model, device)

    assert pred_class == 1

def test_tokenizer_called_once(mocker):
    device = torch.device("cpu")
    fake = mocker.patch.object(inference, "tokenizer", return_value = {"input_ids": torch.tensor([[1,2,3,4]], dtype=torch.long)})

    model = DummyModel(torch.tensor([[1.0,3.0]], dtype=torch.float32))

    inference.predict("text", model, device)

    fake.assert_called_once()

def test_tokenizer_called_once_with_args(mocker):
    device = torch.device("cpu")
    fake = mocker.patch.object(inference, "tokenizer", return_value={"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)})

    model = DummyModel(torch.tensor([[1.0,3.0]], dtype=torch.float32))

    inference.predict("text", model, device)

    fake.assert_called_with("text", padding=True, truncation=True, return_tensors="pt")

def test_predict_calls_model_once_with_input_ids(mocker):
    device = torch.device("cpu")

    mocker.patch.object(
        inference,
        "tokenizer",
        return_value={"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}
    )

    model_mock = mocker.Mock(return_value=torch.tensor([[1.0, 3.0]], dtype=torch.float32))

    inference.predict("text", model_mock, device)

    model_mock.assert_called_once()
    
    (called_input_ids,), _ = model_mock.call_args
    assert isinstance(called_input_ids, torch.Tensor)
    assert called_input_ids.dtype == torch.long
    assert called_input_ids.shape == (1, 4)
    assert called_input_ids.device.type == "cpu"

def test_predict_raises_if_tokenizer_missing_input_ids(monkeypatch):
    device = torch.device("cpu")
    def bad_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"wrong_key": torch.tensor([[1,2,3]])}

    monkeypatch.setattr(inference, "tokenizer", bad_tokenizer)

    model = DummyModel(torch.tensor([[1.0,3.0]]))

    with pytest.raises(KeyError):
        inference.predict("text", model, device)

class BadModel(torch.nn.Module):
    def forward(self, input_ids):
        return torch.tensor([1.0, 3.0])  
    
def test_predict_with_invalid_model_output(monkeypatch):
    device = torch.device("cpu")
    def fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1,2,3]])}

    monkeypatch.setattr(inference, "tokenizer", fake_tokenizer)

    model = BadModel()

    with pytest.raises(Exception):
        inference.predict("text", model, device)

def test_predict_with_empty_input(monkeypatch):
    device = torch.device("cpu")
    def fake_tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": torch.empty((1,0), dtype=torch.long)}
    monkeypatch.setattr(inference, "tokenizer", fake_tokenizer)
    model = DummyModel(torch.tensor([[1.0,3.0]]))

    pred_class, confidence = inference.predict("text", model, device)

    assert pred_class in [0,1]