
"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned to this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab. This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning
is prohibited and subject to being investigated as a Georgia Tech honor code violation.

-----do not edit anything above this line---
"""

import random
import numpy as np
import torch
from torch import nn


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer that encodes a sequence
    and predicts vocabulary-token scores.
    """

    def __init__(
        self,
        input_size,
        output_size,
        device,
        hidden_dim=128,
        num_heads=2,
        dim_feedforward=2048,
        dim_k=96,
        dim_v=96,
        dim_q=96,
        max_length=43,
    ):
        super(TransformerTranslator, self).__init__()

        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        seed_torch(0)

        # Word embeddings
        self.embeddingL = nn.Embedding(
            max(self.input_size, 1586),
            self.hidden_dim
        )

        # Positional embeddings: support at least 1,000 positions
        self.posembeddingL = nn.Embedding(
            max(self.max_length, 43, 1000),
            self.hidden_dim
        )

        # Multi-head self-attention projections
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)

        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)

        self.attention_head_projection = nn.Linear(
            self.dim_v * self.num_heads,
            self.hidden_dim
        )

        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        # Feed-forward network
        self.linear1 = nn.Linear(
            self.hidden_dim,
            self.dim_feedforward
        )

        self.linear2 = nn.Linear(
            self.dim_feedforward,
            self.hidden_dim
        )

        self.norm_ff = nn.LayerNorm(self.hidden_dim)

        # Predict a score for every vocabulary token
        self.classification_head = nn.Linear(
            self.hidden_dim,
            self.output_size
        )

    def embed(self, inputs):
        """Embed input tokens and add positional embeddings."""

        positions = torch.arange(
            inputs.shape[1],
            device=inputs.device
        ).unsqueeze(0)

        word_embeddings = self.embeddingL(inputs)
        position_embeddings = self.posembeddingL(positions)

        return word_embeddings + position_embeddings

    def multi_head_attention(self, inputs):
        """Apply multi-head self-attention."""

        # Attention head 1
        Q1 = self.q1(inputs)
        K1 = self.k1(inputs)
        V1 = self.v1(inputs)

        scores1 = torch.matmul(
            Q1,
            K1.transpose(-2, -1)
        )
        scores1 = scores1 / (self.dim_k ** 0.5)

        attention1 = torch.softmax(scores1, dim=-1)
        head1 = torch.matmul(attention1, V1)

        # Attention head 2
        Q2 = self.q2(inputs)
        K2 = self.k2(inputs)
        V2 = self.v2(inputs)

        scores2 = torch.matmul(
            Q2,
            K2.transpose(-2, -1)
        )
        scores2 = scores2 / (self.dim_k ** 0.5)

        attention2 = torch.softmax(scores2, dim=-1)
        head2 = torch.matmul(attention2, V2)

        # Concatenate and project attention heads
        combined = torch.cat(
            (head1, head2),
            dim=-1
        )

        projected = self.attention_head_projection(combined)

        # Residual connection and layer normalization
        return self.norm_mh(projected + inputs)

    def feedforward_layer(self, inputs):
        """Apply the feed-forward network."""

        outputs = self.linear1(inputs)
        outputs = torch.relu(outputs)
        outputs = self.linear2(outputs)

        return self.norm_ff(inputs + outputs)

    def final_layer(self, outputs):
        """
        Produce vocabulary-token scores for every sequence position.

        Input shape:  (batch_size, sequence_length, hidden_dim)
        Output shape: (batch_size, sequence_length, output_size)
        """

        return self.classification_head(outputs)

    def forward(self, inputs):
        """Run the complete Transformer forward pass."""

        embeddings = self.embed(inputs)

        attention_outputs = self.multi_head_attention(
            embeddings
        )

        feedforward_outputs = self.feedforward_layer(
            attention_outputs
        )

        scores = self.final_layer(
            feedforward_outputs
        )

        return scores


# http://localhost:8888/notebooks/Machine_Translation.ipynb#4.6:-Train-the-Transformer
# The above link is there to check the Jupyter notebook
