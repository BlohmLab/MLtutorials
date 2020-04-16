"""collection of classes and functions for vanilla rnn tutorial"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class vrnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(vrnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # define model layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """perform a forward pass of the model with input x"""

        n_seq = x.size(1) # aka number of batches
        h = self.init_hidden_state(n_seq) # current state

        self.h_all, h = self.rnn(x, h) # h_all stores h for t=0:seq_length

        y = self.out(self.h_all)

        return y, h

    def init_hidden_state(self, n_seq):
        """initialize hidden state of network to zeros"""
        h = torch.zeros(self.num_layers, n_seq, self.hidden_size)
        return h

def generate_input_target(input_size, seq_len, n_seq):
    """generates random noise sequence x and integration target t"""
    
    # sampling from the "standard normal" ~N(0, 1)
    x = np.random.randn(seq_len, n_seq, input_size)
    t = np.cumsum(x, axis=0)

    return x, t

def plot_integration(x, t, y=None, h=None, n_to_plot=5):
    """plots inputs x and integration targets t"""
    plt.figure(figsize=(10, 7))
    
    plt.subplot(311)
    plt.plot(x[:,:min(5, x.shape[1]),0])
    plt.axhline(color='k')
    plt.xlim([0, x.shape[0]-1])
    plt.title("x: (input) random samples from N(0, 1)")
    
    plt.subplot(312)
    plt.plot(t[:,:min(5, x.shape[1]),0])
    plt.axhline(color='k')
    plt.xlim([0, x.shape[0]-1])
    plt.title("t: (target) integral of x")
    
    if y is not None:
        y = y.detach().numpy() # needed bc of grad tracking
        plt.subplot(313)
        plt.plot(y[:,:min(5, x.shape[1]),0])
        plt.axhline(color='k')
        plt.xlim([0, x.shape[0]-1])
        plt.title("y: (output) rnn output")
    
    plt.tight_layout()
    plt.show()

def plot_weights(model):
    """plots recurrent, input, & output weights + biases of pytorch model"""
    plt.figure(figsize=(10, 7))

    plt.subplot(231)
    W_i = model.rnn.weight_ih_l0.detach().numpy()
    plt.stem(W_i, use_line_collection=True)
    plt.title('W_i: input weights')

    plt.subplot(232)
    W_r = model.rnn.weight_hh_l0.detach().numpy()
    plt.imshow(W_r, interpolation=None)
    plt.colorbar()
    plt.title('W_r: reccurrent weights')

    plt.subplot(233)
    W_o = np.squeeze(model.out.weight.detach().numpy())
    plt.stem(W_o, use_line_collection=True)
    plt.title('W_o: output weights')

    plt.subplot(234)
    b_i = model.rnn.bias_ih_l0.detach().numpy()
    plt.stem(b_i, use_line_collection=True)
    plt.title('b_i: input bias')

    plt.subplot(235)
    b_r = model.rnn.bias_hh_l0.detach().numpy()
    plt.stem(b_r, use_line_collection=True)
    plt.title('b_r: recurrent bias')

    plt.subplot(236)
    b_o = model.out.bias.detach().numpy()
    plt.stem(b_o, use_line_collection=True)
    plt.title('b_o: output bias')

    plt.show()
