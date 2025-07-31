# RNN and LSTM Text Classification with PyTorch

This project implements RNN and LSTM models for sentiment classification on the IMDb dataset.

## Project structure

- `models.py` - contains RNN and LSTM model definitions  
- `utils.py` - utility functions for tokenization 
- `train.py` - script to train the models with arguments to choose RNN or LSTM
- `inference.py` - script to run inference (predict sentiment) on new input text   
- `requirements.txt` - Python dependencies

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```
2. Train the model (example for RNN):

```
python train.py --model rnn
```
3. Run inference on new text:
```
python inference.py --model rnn --text "This movie was fantastic!"
```

## Requirements
- Python 3.8+
- PyTorch
- transformers
- datasets

## Notes
- Trained models and logs are saved in models/ and logs/ folders.
- Supports GPU if available, else runs on CPU.
- Modify parameters in train.py for batch size, epochs, etc.
- The inference script supports both rnn and lstm models; specify model type accordingly.

