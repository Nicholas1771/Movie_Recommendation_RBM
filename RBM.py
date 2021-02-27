# Import the libraries
import torch


class RBM():
    # Initialize the random weights, bias
    def __init__(self, nv, nh):
        # Weights
        self.W = torch.randn(nh, nv)

        # Bias of the hidden nodes. Bias of probabilities p_h_given_v
        self.a = torch.randn(1, nh)

        # Bias of the visible nodes. Bias of probabilities p_v_given_h
        self.b = torch.randn(1, nv)

    # Return probabilities of the hidden nodes given x (visible nodes)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # Return probabilities of the visible nodes given y (hidden nodes)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # Train the weights
    # v0 - input vector containing ratings of movies from 1 user
    # vk - visible nodes obtained after k iterations
    # ph0 - vector of probabilities that the hidden nodes = 1 given the values of v0
    # phk - vector of probabilities that the hidden nodes = 1 after k iterations
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
