import torch
from torch import nn

# Model
class NNLM(nn.Module):
    def __init__(self, n_class, m, n_step, n_hidden):
        super(NNLM, self).__init__()
        self.n_step = n_step
        self.m = m

        self.C = nn.Embedding(n_class, m)
        self.w1 = nn.Linear(n_step * m, n_hidden, bias=False)
        self.b1 = nn.Parameter(torch.ones(n_hidden))
        self.w2 = nn.Linear(n_hidden, n_class, bias=False)
        self.w3 = nn.Linear(n_step * m, n_class, bias=False)

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.m)    # X
        Y1 = torch.tanh(self.b1 + self.w1(X)) #Y1 b1 w1
        b2 = self.w3(X)  #b2  W2
        Y2 = b2 + self.w2(Y1) #Y2  #为什么不用加softmax
        return Y2