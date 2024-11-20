import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, 5)  # Output layer for 5 classes
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax for probabilities
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        """
        Forward pass for the Recurrent Neural Network (RNN).

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Log probabilities for each class of shape (batch_size, num_classes).
        """
        _, hidden = self.rnn(inputs)  # Forward pass through RNN, get the final hidden state
        output_representation = self.fc(hidden[-1])  # Final hidden state passed to fully connected layer
        predicted_vector = self.softmax(output_representation)  # Convert logits to log probabilities
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    return tra, val


if __name__ == "__main__":
    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    # Load pre-trained word embeddings
    print("========== Loading word embeddings ==========")
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Ensure '<UNK>' token exists in the word_embedding dictionary
    if unk not in word_embedding:
        word_embedding[unk] = np.zeros(len(next(iter(word_embedding.values()))))  # Initialize '<UNK>' with zeros

    # Initialize the model and optimizer
    embedding_dim = len(next(iter(word_embedding.values())))  # Infer embedding size from the first embedding vector
    model = RNN(embedding_dim, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_validation_accuracy = 0

    print("========== Training ==========")
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)
        loss_total = 0

        # Training loop
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding.get(word.lower(), word_embedding[unk]) for word in input_words]
                vectors = torch.tensor(vectors, dtype=torch.float32).view(1, len(vectors), -1)  # Ensure float32 type
                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        training_accuracy = correct / total
        print(f"Epoch {epoch + 1}: Training Loss = {loss_total / (N // minibatch_size):.4f}, Training Accuracy = {training_accuracy:.4f}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding.get(word.lower(), word_embedding[unk]) for word in input_words]
                vectors = torch.tensor(vectors, dtype=torch.float32).view(1, len(vectors), -1)  # Ensure float32 type
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
        validation_accuracy = correct / total
        print(f"Epoch {epoch + 1}: Validation Accuracy = {validation_accuracy:.4f}")

        # Save the best model
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), "best_rnn_model.pt")
            print(f"New best model saved with validation accuracy: {best_validation_accuracy:.4f}")

    print("Training complete. Best validation accuracy: {:.4f}".format(best_validation_accuracy))
