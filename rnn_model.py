# import torch.nn as nn
# import torch

# class ChatbotRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
#         super(ChatbotRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, input, mask):
#         embedded = self.embedding(input)
#         output, _ = self.rnn(embedded)

#         # Apply the mask to the output to keep only the non-padded values
#         output = output * mask.unsqueeze(2)

#         output = self.fc(output)

#         return output

import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class ChatbotRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob,):
        super(ChatbotRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

        # Set requires_grad=True for all parameters
        # for param in self.parameters():
        #     param.requires_grad_(True)
    
    def forward(self, input, mask):
        embedded = self.embedding(input)

        # Pass the converted tensor to the LSTM module
        output, _ = self.rnn(embedded)

        # Apply the mask to the output to keep only the non-padded values
        output = output * mask.unsqueeze(2)

        # Pad the output back to the original sequence length
        max_seq_length = mask.size(1)
        output_padded = nn.functional.pad(output, pad=(0, max_seq_length - output.size(1)))


        output = self.fc(output_padded)

        return output
