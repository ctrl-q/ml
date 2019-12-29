# Language model code submitted as Representation Learning class assignment

import torch
import torch.nn as nn
from torch._jit_internal import weak_module

import numpy as np
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1


class RecurrentLayer(nn.Module):
    def __init__(self, input_size, output_size, dp_keep_prob):
        super(RecurrentLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear2 = nn.Linear(self.output_size, self.output_size)
        self.dropout = nn.Dropout(1 - dp_keep_prob)

    def init_weights_uniform(self):
        torch.nn.init.uniform_(
            self.linear1.weight, -np.sqrt(1 / self.output_size), np.sqrt(1 / self.output_size))
        torch.nn.init.uniform_(
            self.linear2.weight, -np.sqrt(1 / self.output_size), np.sqrt(1 / self.output_size))
        torch.nn.init.uniform_(
            self.linear2.bias, -np.sqrt(1 / self.output_size), np.sqrt(1 / self.output_size))

    def forward(self, w_x, w_h):
        w_x = self.dropout(w_x)
        w_x = self.linear1(w_x)
        w_h = self.linear2(w_h)
        return torch.tanh(w_x + w_h)


# Implement a stacked vanilla RNN with Tanh nonlinearities.
class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The numvwe of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob  # !

        self.emb_size = emb_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

        self.input_layer = RecurrentLayer(
            emb_size, hidden_size, self.dp_keep_prob)
        self.recur_layer = RecurrentLayer(
            hidden_size, hidden_size, self.dp_keep_prob)
        self.recur_layer = clones(self.recur_layer, self.num_layers - 1)
        self.recur_layer.insert(0, self.input_layer)
        self.dropout = nn.Dropout(1 - dp_keep_prob)

        self.out_layer = nn.Linear(
            self.hidden_size, self.vocab_size, bias=True)

        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)

        # init embedding weights
        torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # init linear output bias and weights
        torch.nn.init.uniform_(self.out_layer.weight, -0.1, 0.1)
        torch.nn.init.zeros_(self.out_layer.bias)

    def init_hidden(self):
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        hidden = torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size)
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return hidden

    def forward(self, inputs, hidden):
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        logits = []

        for tstep in inputs:

            for layer in range(self.num_layers):

                if layer < 1:
                    embeds = self.embedding(tstep)
                else:
                    embeds = hidden[layer - 1].clone()

                hidden_layer = hidden[layer].clone()
                hidden[layer] = self.recur_layer[layer](embeds, hidden_layer)

            outs = self.dropout(hidden[-1].clone())
            outs = self.out_layer(outs)
            logits.append(outs)

        logits = torch.stack(logits)

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        samples = torch.zeros(
            [generated_seq_len, self.batch_size], device=input.device)

        samples = []

        for i in generated_seq_len:

            logits, hidden = self(input, hidden)
            logits = nn.Softmax(logits)
            input = torch.argmax(logits)
            samples.append(input)

        return samples


# Problem 2
class OutputLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size, p):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights_uniform()
        self.dropout = nn.Dropout(p)

    def init_weights_uniform(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.dropout(x)
        out = self.fc(x)
        return out


class Gate(nn.Module):
    def __init__(self, input_dim, hidden_dim, p, activation_function='tanh'):
        super(Gate, self).__init__()
        assert(activation_function in ['sigmoid', 'tanh'])
        if activation_function == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.activation = nn.Tanh()

        self.p = p
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=self.p)
        self.k = np.sqrt(1 / hidden_dim)
        self.init_weights_uniform()

    def init_weights_uniform(self):
        nn.init.uniform_(self.linear1.weight, a=-self.k, b=self.k)
        nn.init.uniform_(self.linear2.weight, a=-self.k, b=self.k)
        nn.init.uniform_(self.linear2.bias, a=-self.k, b=self.k)

    def forward(self, x, h):
        x = self.dropout(x)
        x = self.linear1(x)
        h = self.linear2(h)
        out = x + h
        out = self.activation(out)
        return out


class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, p):
        super(GRULayer, self).__init__()

        self.p = p
        self.dropout = nn.Dropout(p=self.p)
        self.r_gate = Gate(input_dim, hidden_dim, p,
                           activation_function='sigmoid')
        self.z_gate = Gate(input_dim, hidden_dim, p,
                           activation_function='sigmoid')
        self.h_gate = Gate(input_dim, hidden_dim, p,
                           activation_function='tanh')

    def forward(self, x, h):
        r = self.r_gate(x, h)
        z = self.z_gate(x, h)
        assert(r.shape == h.shape)
        assert (z.shape == h.shape)
        h_t = self.h_gate(x, r * h)
        assert (h_t.shape == h.shape)
        h = (1 - z) * h + z * h_t
        return h


class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.p = 1 - dp_keep_prob
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        self.input_layer = GRULayer(emb_size, hidden_size, p=self.p)
        self.gru_layer = GRULayer(hidden_size, hidden_size, p=self.p)
        self.output_layer = OutputLayer(
            self.hidden_size, self.vocab_size, p=self.p)

        self.gru_layers = clones(self.gru_layer, self.num_layers - 1)
        self.gru_layers.insert(0, self.input_layer)
        self.init_weights_uniform()

    def init_weights_uniform(self):
        nn.init.uniform_(self.embeddings.weight, a=-0.1, b=0.1)

    def init_hidden(self):
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        h = torch.zeros([self.num_layers, self.batch_size, self.hidden_size])
        if torch.cuda.is_available():
            h = h.cuda()
        return h

    def forward(self, inputs, hidden):
        logits = torch.zeros(
            [self.seq_len, self.batch_size, self.vocab_size], device=inputs.device)
        C = self.embeddings(inputs)
        C = C.view(self.seq_len, -1, self.emb_size)
        for t in range(self.seq_len):
            x = C[t]
            h = []
            for layer in range(self.num_layers):
                temp = self.gru_layers[layer](x, hidden[layer])
                h.append(temp)
                x = temp
            hidden = torch.stack(h)
            logits[t] = self.output_layer(x)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        samples = torch.zeros(
            [generated_seq_len, self.batch_size], device=input.device)
        for i in generated_seq_len:
            logits, hidden = self(input, hidden)
            input = torch.argmax(nn.Softmax(logits))
            samples[i] = input

        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""

# This code has been modified from an open-source project, by David Krueger.
# The original license is included below:
# MIT License
#
# Copyright (c) 2018 Alexander Rush
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ----------------------------------------------------------------------------------
@weak_module
class LinearU(nn.Linear):
    def reset_parameters_uniform(self):
        """Initializes layer from U[-k, k], k == 1 / sqrt(self.n_units)"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_heads = n_heads
        self.n_units = n_units
        self.dropout = nn.Dropout(p=dropout)

        # since d_k = d_v = n_units // n_heads, we can use n_units as the 2nd dimension
        # for all linear layers and init them in one line
        # V, K, Q & output layers
        self.linear_layers = clones(LinearU(n_units, n_units), 4)

    def forward(self, query, key, value, mask=None):
        # query, key, and value all have size: (batch_size, seq_len, self.n_units, self.d_k)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        if mask is not None:
            mask = mask.unsqueeze(1)

        q_k_v = []
        for layer, val in zip(self.linear_layers, [query, key, value]):
            tensor = layer(val).view(query.size(0), -1, self.n_heads, self.d_k)
            tensor = tensor.transpose(1, 2)
            q_k_v.append(tensor)

        Q, K, V = q_k_v

        # Calculate attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        softmax = self.dropout(F.softmax(scores, dim=-1))
        attention = torch.matmul(softmax, V)

        # Concatenate
        attention = attention.transpose(1, 2).contiguous().view(
            query.size(0), -1, self.n_units)
        out = self.linear_layers[-1]
        return out(attention)


# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(
            x, x, x, mask))  # apply the self-attention
        # apply the position-wise MLP
        return self.sublayer[1](x, self.feed_forward)


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(
            n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(
            n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
