from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_texts(texts):
    
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

def tokenize_function(example):
    return tokenizer(example['text'], padding = 'max_length', truncation=True, max_length=128)