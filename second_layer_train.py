import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import time
import torch
from second_layer_model import GCN_Model
from second_layer_prepare_data import create_graph, add_feature, create_dataloader


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return np.round(correct / len(labels), 2)

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epoch, eval=True):
    f = open('Output/color_change_result.txt', 'w')
    
    t = time.time()
    loss_hist = []
    for _ in range(epoch):
        model.train()
        for features, adj_matrix, labels in iter(train_dataloader):
            features = features.squeeze()
            adj_matrix = adj_matrix.squeeze()
            labels = labels.squeeze()
            
            output = model(features, adj_matrix)
            # 挑出label == 996的，不參與loss計算
            excluded_index = []
            for i in range(len(labels)):
                if(labels[i] != 996):
                    excluded_index.append(i)
            if(excluded_index):
                real_labels = torch.index_select(labels, dim=0, index=torch.tensor(excluded_index))
                real_output = torch.index_select(output, dim=0, index=torch.tensor(excluded_index))
                
                loss = loss_fn(real_output, real_labels)
                loss_hist.append(float(loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if(eval):
            model.eval()
            total_loss = 0
            for features, adj_matrix, labels in iter(test_dataloader):
                features = features.squeeze()
                adj_matrix = adj_matrix.squeeze()
                labels = labels.squeeze()
                output = model(features, adj_matrix)
                # 挑出label == 996的，不參與loss計算
                excluded_index = []
                for i in range(len(labels)):
                    if(labels[i] != 996):
                        excluded_index.append(i)
                if(excluded_index):
                    real_labels = torch.index_select(labels, dim=0, index=torch.tensor(excluded_index))
                    # print(len(real_labels))
                    real_output = torch.index_select(output, dim=0, index=torch.tensor(excluded_index))
                    total_loss += float(loss_fn(real_output, real_labels))
            f.write('eval avg loss: ' + str(total_loss/len(test_dataloader)) + '\n')
            f.write('acc: ' + str(float(accuracy(real_output, real_labels))) + '\n')
            f.write('='*80 + '\n\n')
    f.close()
    return loss_hist

if __name__ == '__main__':
    feature_df = pd.read_csv('data/change_color_df.csv')
    structure_df = pd.read_csv('data/graph_structure.csv')
    col = feature_df.columns.tolist()
    col.remove('color')
    new_col = ['color_red', 'color_black', 'color_green', 'color_blue'] + col

    ct = ColumnTransformer([('color', OneHotEncoder(), [3])], remainder='passthrough')
    feature_onehot = np.array(ct.fit_transform(feature_df))
    feature_onehot_df = pd.DataFrame(feature_onehot, columns=new_col)

    graph_list = []
    label_list = []
    for session in range(1, 14):
        s_all_df = structure_df[structure_df['session'] == session]
        for round in s_all_df['round'].unique():
            
            s_df = s_all_df[s_all_df['round'] == round]
            f_df = feature_onehot_df.iloc[s_df.index]
            graph = create_graph(s_df)
            graph = add_feature(graph, f_df)
            graph_list.append(graph)

            labels = f_df['new_color'].tolist()
            label_list.append(labels)

    train_dataloader, test_dataloader = create_dataloader(graph_list, label_list)

    model = GCN_Model(9, 9, 4, 0.2)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    l = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epoch=20)