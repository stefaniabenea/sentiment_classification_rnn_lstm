import torch
from models import LSTMClassifier, RNNClassifier
import pytest

@pytest.mark.parametrize("batch_size",[1,8])
def test_lstm_forward_output_shape(batch_size):
    # Arrange
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1

    # Act
    model = LSTMClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x = torch.randint(0,vocab_size, (batch_size, 10))
    output = model(x)
    # Assert
    assert output.shape == (batch_size, num_classes)

def test_lstm_forward_is_float_tensor():
   # Arrange
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1

    #Act
    model = LSTMClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x = torch.randint(0,vocab_size, (3, 5))
    output = model(x)

    #Assert
    assert output.dtype in (torch.float32, torch.float64) 

@pytest.mark.parametrize("batch_size",[1,8])
def test_rnn_forward_output_shape(batch_size):
     # Arrange
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1

    #Act
    model = RNNClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x= torch.randint(0, vocab_size,(batch_size, 10))
    output = model(x)

    #Assert
    assert output.shape == (batch_size, num_classes)

def test_rnn_forward_is_float_tensor():
   # Arrange
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1

    #Act
    model = RNNClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x = torch.randint(0,vocab_size, (3, 5))
    output = model(x)

    #Assert
    assert output.dtype in (torch.float32, torch.float64) 

@pytest.mark.parametrize("sequence_length",[1,100])
def test_lstm_forward_output_shape(sequence_length):
    # Arrange
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1

    # Act
    model = LSTMClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x = torch.randint(0,vocab_size, (4, sequence_length))
    output = model(x)
    # Assert
    assert output.shape == (4, num_classes)

@pytest.mark.parametrize("sequence_length",[1,100])
def test_rnn_forward_output_shape(sequence_length):
    # Arrange
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1

    # Act
    model = RNNClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x = torch.randint(0,vocab_size, (4, sequence_length))
    output = model(x)
    # Assert
    assert output.shape == (4, num_classes)

def test_lstm_deterministic_in_eval():
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1
    model = LSTMClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    model.eval()
    x = torch.randint(0,vocab_size, (4, 10))
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)

def test_rnn_deterministic_in_eval():
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1
    model = RNNClassifier(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    model.eval()
    x = torch.randint(0,vocab_size, (4, 10))
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)

@pytest.mark.parametrize("modelclass",[RNNClassifier, LSTMClassifier])
@pytest.mark.parametrize("device",["cpu","cuda"])
def test_output_on_same_device_cpu(modelclass, device):
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1
    model = modelclass(vocab_size, embed_size, hidden_dim, num_classes, num_layers)

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    x = torch.randint(0,vocab_size, (4, 10), device = device)
    out = model(x)
    assert out.device == device

@pytest.mark.parametrize("modelclass",[RNNClassifier,LSTMClassifier])
def test_raises_on_float_input_dtype(modelclass):
    vocab_size = 100
    embed_size = 32
    hidden_dim = 64
    num_classes = 2
    num_layers = 1
    model = modelclass(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    x = torch.randint(0,vocab_size, (4, 10), dtype=torch.float32)
    with pytest.raises((RuntimeError, TypeError)):
        model(x)

@pytest.mark.parametrize("model_cls", [RNNClassifier, LSTMClassifier])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (2, 5), (4, 10)])
def test_internal_tensor_shapes(model_cls, batch_size, seq_len):
    vocab_size = 100
    embed_size = 16
    hidden_dim = 32
    num_classes = 2
    num_layers = 2  

    model = model_cls(vocab_size, embed_size, hidden_dim, num_classes, num_layers)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    # (B,L) -> (B, L, embed_size)
    embedded = model.embedding(x)
    assert embedded.shape == (batch_size, seq_len, embed_size)

    if hasattr(model,'rnn'):
        # output: (B,L,hidden_dim)
        # hidden: (num_layers, B, hidden_dim)
        output, hidden = model.rnn(embedded)
        assert output.shape == (batch_size,seq_len,hidden_dim)
        assert hidden.shape == (num_layers, batch_size, hidden_dim)
        last_hidden_layer = hidden[-1]
        assert last_hidden_layer.shape == (batch_size, hidden_dim)
    elif hasattr(model, "lstm"):
        # output: (B,L,hidden_dim)
        # hidden: (num_layers, B, hidden_dim)
        output, (hidden, cell) = model.lstm(embedded)
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert hidden.shape == (num_layers, batch_size, hidden_dim)
        assert cell.shape == (num_layers, batch_size, hidden_dim)
        last_hidden_layer = hidden[-1]
        assert last_hidden_layer.shape == (batch_size, hidden_dim)
    else:
        raise AssertionError("Model must have either .rnn or .lstm atttribute")
    
    logits = model.fc(last_hidden_layer)
    assert logits.shape == (batch_size, num_classes)
