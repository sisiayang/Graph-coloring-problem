import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader

class graphDataset(Dataset):
    def __init__(self, graph_list, label_list):
        feature_matrix_list = []
        for g in graph_list:
            feature_matrix = [list(g.nodes[j].values()) for j in range(1, g.number_of_nodes()+1)]
            feature_matrix_list.append(torch.tensor(feature_matrix, dtype=torch.float))
        
        self.feature_matrix = feature_matrix_list
        
        adj_matrix_list = []
        for g in graph_list:
            adj_matrix = nx.adjacency_matrix(g).todense()
            adj_matrix_list.append(torch.tensor(adj_matrix, dtype=torch.float))
        self.adj_matrix = adj_matrix_list

        labels = []
        for i in range(len(label_list)):
            labels.append(torch.tensor(label_list[i], dtype=torch.int64))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.feature_matrix[idx]
        adj_matrix = self.adj_matrix[idx]
        labels = self.labels[idx]

        return features, adj_matrix, labels

def create_graph(df):  # df size = graph size
    size = len(df)+1
    G = nx.Graph()
    nodes = range(1, size)
    G.add_nodes_from(nodes)
    edge_list = []
    for uid in range(1, size):
        neighbor_list = df[df['uid'] == uid].iloc[0]['linked'].split(',')
        edge_list = [(int(uid), int(neighbor)) for neighbor in neighbor_list]
        G.add_edges_from(edge_list)
        G.add_edge(int(uid), int(uid))
    return G

def add_feature(graph, f_df):
    feature_col_list = ['color_red', 'color_black', 'color_green', 'color_blue', 'score', 'num_of_neighbor', 'hist_color', 'hist_neighbor', 'hist_skip']
    for n in graph.nodes():
        for f in feature_col_list:
            graph.nodes[n][f] = f_df[f_df['uid']==n][f].tolist()[0]    # 取出uid符合的user的f欄位，從series型態轉成list再取值
    return graph

def create_dataloader(graph_list, label_list):
    train_dataset = graphDataset(graph_list[:90], label_list[:90])
    test_dataset = graphDataset(graph_list[90:], label_list[90:])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_dataloader, test_dataloader