import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GraphConvolutionalLayer(nn.modules.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GraphConvolutionalLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.bias = None
    def forward(self, adj_matrix, features):
        output = torch.mm(adj_matrix, features)
        output = torch.mm(output, self.weight)
        if(self.bias is not None):
            return output + self.bias
        else:
            return output
    
class GCN_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GCN_Model, self).__init__()
        self.gcn = GraphConvolutionalLayer(in_dim, hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        
    def forward(self, feature, adj_matrix):
        x = torch.nn.functional.relu(self.gcn(adj_matrix, feature))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output