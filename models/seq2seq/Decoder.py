
"""
S2S Decoder model. (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """The Decoder module of the Seq2Seq model."""

    def __init__(
        self,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        output_size,
        dropout=0.2,
        model_type="RNN",
        attention=False
    ):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        # 1. Embedding
        self.embedding = nn.Embedding(output_size, emb_size)

        # 2. Recurrent layer
        recurrent_input_size = emb_size
        if attention:
            recurrent_input_size += encoder_hidden_size

        if model_type == "RNN":
            self.rnn = nn.RNN(
                recurrent_input_size,
                decoder_hidden_size,
                batch_first=True
            )
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                recurrent_input_size,
                decoder_hidden_size,
                batch_first=True
            )
        else:
            raise ValueError("model_type must be 'RNN' or 'LSTM'")

        # 3. Output projection and log probabilities
        self.fc_out = nn.Linear(decoder_hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # 4. Dropout
        self.dropout = nn.Dropout(dropout)

        # 5. Attention projection
        if attention:
            self.attn_combine = nn.Linear(
                encoder_hidden_size + emb_size, emb_size
            )
            # Project encoder features if their size differs from
            # the decoder hidden size.
            self.attn_key = nn.Linear(
                encoder_hidden_size, decoder_hidden_size, bias=False
            )

    def compute_attention(self, hidden, encoder_outputs):
        """Calculate cosine-similarity attention.

        hidden: (1, batch, decoder_hidden_size)
        encoder_outputs: (batch, seq_len, encoder_hidden_size)
        returns: (batch, 1, seq_len)
        """
        query = hidden[-1].unsqueeze(1)

        # Align dimensions if encoder and decoder hidden sizes differ.
        if query.size(-1) != encoder_outputs.size(-1):
            keys = self.attn_key(encoder_outputs)
        else:
            keys = encoder_outputs

        query = nn.functional.normalize(query, p=2, dim=-1)
        keys = nn.functional.normalize(keys, p=2, dim=-1)

        scores = torch.bmm(query, keys.transpose(1, 2))
        attention_prob = torch.softmax(scores, dim=-1)

        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """Run one decoder time step."""

        embedded = self.dropout(self.embedding(input))

        if self.attention:
            if encoder_outputs is None:
                raise ValueError(
                    "encoder_outputs is required when attention=True"
                )

            # Use the hidden state as the attention query.
            query_hidden = hidden[0] if self.model_type == "LSTM" else hidden
            attention_prob = self.compute_attention(
                query_hidden, encoder_outputs
            )

            context = torch.bmm(attention_prob, encoder_outputs)

            # Concatenate context with the embedded decoder input.
            #combined = torch.cat((context, embedded), dim=-1)
            #embedded = torch.tanh(self.attn_combine(combined))
            combined = torch.cat((context, embedded), dim=-1)
            embedded = combined

        output, hidden = self.rnn(embedded, hidden)

        output = self.fc_out(output.squeeze(1))
        output = self.log_softmax(output)

        return output, hidden
