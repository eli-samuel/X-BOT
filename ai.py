import os
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import defaultdict
import codecs
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets, transforms
import torchvision
import torch
from torch.nn.modules.activation import Softmax
from torchsummary import summary
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
torch.set_printoptions(threshold=5000)

from functions import write_to_file, read_from_file, add_masking_padding
import copy
from rnn_model import ChatbotRNN
from tqdm import tqdm
import time

# Read the contents of the files containing the training data, target data, and names data
with tqdm(total=1, desc="Importing files") as pbar:
    training = read_from_file("training\\trainingALL.txt")
    target = copy.deepcopy(training)

    target = target[1:]  # Remove the first element
    target.append("")  # Append a last element to make it the same length as training

    # print(len(training))
    # print(len(target))
    # print("Done importing files!\n")

    pbar.update(1)


# Tokenization
import nltk

# Download NLTK resources (only needed once)
# nltk.download('punkt')

tokenized_messages = []

with tqdm(total=len(training), desc="Tokenizing messages") as pbar:
    # Tokenize each text message
    for message in training:
        tokenized_message = nltk.word_tokenize(message)
        tokenized_messages.append(tokenized_message)
        pbar.update(1)

# Removing Stopwords
from nltk.corpus import stopwords

# Download stopwords (only needed once)
# nltk.download('stopwords')

filtered_messages = []

# Remove stopwords from tokenized messages
stopwords = set(stopwords.words('english'))
with tqdm(total=len(tokenized_messages), desc="Filtering messages") as pbar:
    for message in tokenized_messages:
        filtered_message = [word for word in message if word.lower() not in stopwords]
        filtered_messages.append(filtered_message)
        pbar.update(1)

    # filtered_messages = [[word for word in message if word.lower() not in stopwords] for message in tokenized_messages]

# print(filtered_messages[0:5])


# Converting to Numerical Representations
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create a tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")

# Fit tokenizer on the filtered messages
tokenizer.fit_on_texts(filtered_messages)


# Filter the word index based on frequency threshold
min_count = 5
tokenizer.word_index = {word: index for word, index in tokenizer.word_index.items() if tokenizer.word_counts.get(word, 0) >= min_count}

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)

# Convert text to sequences of token indices
sequences = tokenizer.texts_to_sequences(filtered_messages)

placeholder_value = 0  # Set your desired placeholder value
sequences_cleaned = [[token if token is not None else placeholder_value for token in sequence] for sequence in sequences]

tokenizer.word_index["<OOV>"] = len(tokenizer.word_index) + 1

# Pad sequences and create a mask
padded_sequences = pad_sequences(sequences_cleaned)
input_with_oov = [[token if token < len(tokenizer.word_index) else tokenizer.word_index["<OOV>"] for token in sequence] for sequence in padded_sequences]

# Create a mask
mask = np.not_equal(input_with_oov, 0).astype(float)

# Convert the mask to a TensorFlow tensor
mask = tf.convert_to_tensor(mask)

# Convert the padded sequences to a TensorFlow tensor
input_with_oov = tf.convert_to_tensor(input_with_oov)

# Create input-output pairs
input_sequences = input_with_oov[:-1]
output_sequences = input_with_oov[1:]

# Create a mask
mask = np.not_equal(input_sequences, 0).astype(float)

# Convert the mask to a TensorFlow tensor
mask = tf.convert_to_tensor(mask)

# Convert input and output sequences to PyTorch tensors
input_sequences = torch.from_numpy(input_sequences.numpy())
output_sequences = torch.from_numpy(output_sequences.numpy())
mask = torch.from_numpy(mask.numpy())

# Calculate the sum of sequences along the second dimension
sequence_sums = input_sequences.sum(dim=1)

# Find the indices of non-zero length sequences
non_zero_length_indices = torch.nonzero(sequence_sums > 0, as_tuple=False).squeeze()

# Remove zero-length sequences from input sequences tensor
input_sequences = input_sequences[non_zero_length_indices]

# Remove index i from mask tensor
mask = mask[non_zero_length_indices]

# Calculate the sum of sequences along the second dimension
sequence_sums = output_sequences.sum(dim=1)

# Find the indices of non-zero length sequences
non_zero_length_indices = torch.nonzero(sequence_sums > 0, as_tuple=False).squeeze()

# Remove corresponding indices from output sequences tensor
output_sequences = output_sequences[non_zero_length_indices]

# Create a TensorDataset
dataset = TensorDataset(input_sequences, output_sequences, mask)

# Define batch size
batch_size = 512

# Create a DataLoader for training data
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Instantiate the Seq2Seq loss function
seq2seq_loss = nn.CrossEntropyLoss()

input_size = vocab_size+1
hidden_size = 100
output_size = vocab_size+1
num_layers = 2
dropout_prob = 0.1
model = ChatbotRNN(input_size, hidden_size, output_size, num_layers, dropout_prob)
model = model.to(torch.float64)

# Instantiate the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (inputs, targets, mask) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        # Forward pass
        outputs = model(inputs, mask)  # Assuming mask is the mask tensor for input sequences

        outputs = outputs.to(torch.float64)
        targets = targets.to(torch.float64)

        # outputs.requires_grad_()
        # targets.requires_grad_()

        # Convert the input tensor to a one-hot representation
        one_hot_target = torch.zeros(targets.size(0), targets.size(1), vocab_size+1, dtype=torch.float64).to(device)
        one_hot_target.scatter_(2, targets.long().unsqueeze(2), 1)


        one_hot_output = torch.zeros(outputs.size(0), outputs.size(1), vocab_size+1, dtype=torch.float64).to(device)
        one_hot_output.scatter_(2, outputs.argmax(dim=2).long().unsqueeze(2), 1)

        one_hot_target.requires_grad_()
        one_hot_output.requires_grad_()

        # Calculate the Seq2Seq loss
        loss = seq2seq_loss(one_hot_output, one_hot_target).to(device)

        # Backward pass and optimization
        optimizer.zero_grad()

        # from torchviz import make_dot
        # # Visualize the computation graph
        # dot = make_dot(outputs, params=dict(model.named_parameters()))
        # dot.render("computation_graph")



        loss.backward()

       


        # if torch.is_grad_enabled():
        #     print("Gradient computation is enabled")
        # else:
        #     print("Gradient computation is disabled")

        # print(dict(model.named_parameters()))

        # # Access the gradients of the model's parameters
        # for name, param in model.named_parameters():
        #     # if param.grad is not None:
        #         print(f"Gradient of parameter {name}:")
        #         print(param.grad)
        #         print(param.is_leaf)
        #         print(param.requires_grad)

        optimizer.step()

        

        total_loss += loss.item()
        print('\tLoss/train', round(total_loss / (i+1), 8), epoch * len(train_dataloader) + (i+1))

    avg_loss = total_loss / len(train_dataloader)
    print(f"Loss after {epoch+1} epochs: {avg_loss}")



torch.save(model.state_dict(), 'model_parameters.pth')

test_model = ChatbotRNN()
test_model.load_state_dict(torch.load('model_parameters.pth'))

print(f"Testing after {epoch+1} epochs with input {tokenizer.sequences_to_texts([inputs[100].tolist()])}")
test_model.eval()
out = test_model(input_sequences[100].unsqueeze(0), mask[100].unsqueeze(0))
out = out.argmax(dim=2)
print(out)
print(tokenizer.sequences_to_texts([out[0].tolist()]))
