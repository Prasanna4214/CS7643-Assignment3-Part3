
import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model. (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """The Sequence to Sequence model."""

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device

        # Move encoder and decoder to the selected device.
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, source):
        """Forward pass of the Seq2Seq model.

        Args:
            source (tensor): source sequences of shape
                             (batch_size, seq_len)
        """
        batch_size = source.shape[0]
        seq_len = source.shape[1]

        # Encode the source sequence.
        encoder_outputs, hidden = self.encoder(source)

        # The first source token is the <sos> token.
        decoder_input = source[:, 0].unsqueeze(1)

        # Initialize output tensor.
        outputs = torch.zeros(
            batch_size,
            seq_len,
            self.decoder.output_size,
            device=source.device
        )

        # Decode one time step at a time.
        for t in range(seq_len):
            if self.decoder.attention:
                output, hidden = self.decoder(
                    decoder_input,
                    hidden,
                    encoder_outputs
                )
            else:
                output, hidden = self.decoder(
                    decoder_input,
                    hidden
                )

            outputs[:, t, :] = output

            # Select the next token using greedy decoding.
            decoder_input = output.argmax(dim=1).unsqueeze(1)

        return outputs
