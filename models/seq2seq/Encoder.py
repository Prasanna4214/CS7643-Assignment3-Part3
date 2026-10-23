
"""
S2S Encoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab. This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """The Encoder module of the Seq2Seq model."""

    def __init__(
        self,
        input_size,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        dropout=0.2,
        model_type="RNN"
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type

        # 1. Embedding layer
        self.embedding = nn.Embedding(input_size, emb_size)

        # 2. Recurrent layer
        if model_type == "RNN":
            self.rnn = nn.RNN(
                input_size=emb_size,
                hidden_size=encoder_hidden_size,
                batch_first=True
            )
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=emb_size,
                hidden_size=encoder_hidden_size,
                batch_first=True
            )
        else:
            raise ValueError("model_type must be 'RNN' or 'LSTM'")

        # 3. Linear -> ReLU -> Linear for hidden-state mapping
        self.fc1 = nn.Linear(
            encoder_hidden_size, encoder_hidden_size
        )
        self.fc2 = nn.Linear(
            encoder_hidden_size, decoder_hidden_size
        )

        # 4. Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """Encode input sequences.

        Args:
            input: Tensor of shape (batch_size, seq_len)

        Returns:
            output: Recurrent outputs for each time step.
            hidden: Final hidden state, or (hidden, cell) for LSTM.
        """

        # Embed input tokens and apply dropout
        embedded = self.dropout(self.embedding(input))

        # Run the recurrent layer
        output, hidden = self.rnn(embedded)

        # Map hidden state to decoder hidden size
        if self.model_type == "LSTM":
            h_n, c_n = hidden

            h_n = torch.tanh(
                self.fc2(torch.relu(self.fc1(h_n)))
            )

            # Do not transform the LSTM cell state
            hidden = (h_n, c_n)

        else:
            hidden = torch.tanh(
                self.fc2(torch.relu(self.fc1(hidden)))
            )

        return output, hidden
