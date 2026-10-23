"""
LSTM model.  (c) 2021 Georgia Tech

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

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """Init function for LSTM class.

        Args:
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO: Declare LSTM weights and attributes.                                    #
        ################################################################################

        # i_t: input gate
        self.Wii = nn.Parameter(torch.empty(input_size, hidden_size))
        self.bii = nn.Parameter(torch.empty(hidden_size))
        self.Whi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bhi = nn.Parameter(torch.empty(hidden_size))

        # f_t: forget gate
        self.Wif = nn.Parameter(torch.empty(input_size, hidden_size))
        self.bif = nn.Parameter(torch.empty(hidden_size))
        self.Whf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bhf = nn.Parameter(torch.empty(hidden_size))

        # g_t: cell gate
        self.Wig = nn.Parameter(torch.empty(input_size, hidden_size))
        self.big = nn.Parameter(torch.empty(hidden_size))
        self.Whg = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bhg = nn.Parameter(torch.empty(hidden_size))

        # o_t: output gate
        self.Wio = nn.Parameter(torch.empty(input_size, hidden_size))
        self.bio = nn.Parameter(torch.empty(hidden_size))
        self.Who = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bho = nn.Parameter(torch.empty(hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)."""

        ################################################################################
        # TODO: Implement the forward pass of LSTM.                                    #
        ################################################################################

        batch_size, seq_len, _ = x.shape

        h_t = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype
        )

        c_t = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype
        )

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Input gate
            i_t = self.sigmoid(
                x_t @ self.Wii + self.bii
                + h_t @ self.Whi + self.bhi
            )

            # Forget gate
            f_t = self.sigmoid(
                x_t @ self.Wif + self.bif
                + h_t @ self.Whf + self.bhf
            )

            # Cell gate
            g_t = self.tanh(
                x_t @ self.Wig + self.big
                + h_t @ self.Whg + self.bhg
            )

            # Output gate
            o_t = self.sigmoid(
                x_t @ self.Wio + self.bio
                + h_t @ self.Who + self.bho
            )

            # Update cell state and hidden state
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return (h_t, c_t)