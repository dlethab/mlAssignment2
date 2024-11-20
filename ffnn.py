import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

# Unknown token for out-of-vocabulary words
unk = '<UNK>'

# Define Feedforward Neural Network (FFNN) model
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h

        # Define layers
        self.W1 = nn.Linear(input_dim, h)        # First hidden layer
        self.fc3 = nn.Linear(h, h // 2)         # Second hidden layer
        self.fc4 = nn.Linear(h // 2, 5)         # Output layer
        self.activation = nn.ReLU()            # Activation function
        self.dropout = nn.Dropout(p=0.3)       # Dropout for regularization
        self.softmax = nn.LogSoftmax(dim=1)    # LogSoftmax for output probabilities
        self.loss = nn.NLLLoss()               # Negative log-likelihood loss

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Ensure input_vector has at least two dimensions
        if len(input_vector.shape) == 1:
            input_vector = input_vector.unsqueeze(0)

        # First hidden layer with activation and dropout
        hidden_representation = self.activation(self.W1(input_vector))
        hidden_representation = self.dropout(hidden_representation)

        # Second hidden layer with activation
        hidden_representation = self.activation(self.fc3(hidden_representation))

        # Output layer
        output_representation = self.fc4(hidden_representation)

        # Log probability distribution
        predicted_vector = self.softmax(output_representation)
        return predicted_vector


# Make vocabulary from training data
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Create word indices and reverse mappings
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {word: idx for idx, word in enumerate(vocab_list)}
    index2word = {idx: word for idx, word in enumerate(vocab_list)}
    vocab.add(unk)
    return vocab, word2index, index2word


# Convert text data to vectorized representation
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


# Load training and validation data
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    return tra, val


# Main function for training
if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Load and process data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Initialize the model and optimizer
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

    # Training loop
    print(f"========== Training for {args.epochs} epochs ==========")
    best_accuracy = 0

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        correct = 0
        total = 0
        random.shuffle(train_data)  # Shuffle data for each epoch
        minibatch_size = 32
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_vector = input_vector.unsqueeze(0)  # Add batch dimension
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        # Report training accuracy
        print(f"Epoch {epoch + 1}: Training accuracy = {correct / total:.4f}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_vector, gold_label in valid_data:
                if len(input_vector.shape) == 1:
                    input_vector = input_vector.unsqueeze(0)
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
        validation_accuracy = correct / total
        print(f"Epoch {epoch + 1}: Validation accuracy = {validation_accuracy:.4f}")

        # Save the best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
