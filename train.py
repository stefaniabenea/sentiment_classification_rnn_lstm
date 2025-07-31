import torch
from utils import tokenize_texts, tokenize_function, tokenizer
from models import RNNClassifier, LSTMClassifier
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import argparse
import csv
import os


os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--model",choices=['rnn','lstm'],required=True)
args = parser.parse_args()
arg_model = args.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data
dataset = load_dataset('imdb')
#tokenized_dataset = dataset.map(tokenize_function,batched=True)
#tokenized_dataset.save_to_disk('tokenized_imdb')
tokenized_dataset = load_from_disk('tokenized_imdb')
tokenized_dataset.set_format(type='torch',columns = ['input_ids','label'])

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True)
test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=16, shuffle=False)

if arg_model == 'rnn':
    model = RNNClassifier(vocab_size=tokenizer.vocab_size, embed_size=100, hidden_dim=64, num_classes=2, num_layers = 1)
elif arg_model == 'lstm':
    model = LSTMClassifier(vocab_size=tokenizer.vocab_size, embed_size=100, hidden_dim=64, num_classes=2, num_layers=1)
else:
    raise ValueError(f"Model {arg_model} not recognized")

model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()


train_losses = []
test_losses =[]
train_accuracies = []
test_accuracies = []
num_epochs = 10

for epoch in range(num_epochs):
    corrects = 0
    losses = 0
    samples = 0 
    model.train()
    for batch in train_dataloader: 
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        losses += loss.item()
        corrects+= (predictions == labels).sum().item()
        samples += labels.size(0)

    avg_loss = losses/len(train_dataloader)
    accuracy = 100*corrects/samples
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    model.eval()
    test_losses_sum = 0
    test_samples = 0
    test_corrects = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            test_losses_sum+=loss.item()
            test_corrects += (predictions==labels).sum().item()
            test_samples+=labels.size(0)
        avg_test_loss = test_losses_sum/len(test_dataloader)
        test_accuracy = 100* test_corrects/test_samples
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}\n train loss: {avg_loss:.4f}, train accuracy: {accuracy:.2f}%, test loss: {avg_test_loss:.4f}, test accuracy: {test_accuracy:.2f}%")
        
with open(f"logs/{arg_model}_training_log.csv", mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])
    for i in range(num_epochs):
        writer.writerow([i+1, train_losses[i], train_accuracies[i], test_losses[i], test_accuracies[i]])

print(f"Training log saved to logs/{arg_model}_training_log.csv")

            
torch.save(model.state_dict(), f'models/{arg_model}.pth')
print(f"Model saved to models/{arg_model}.pth")



        

