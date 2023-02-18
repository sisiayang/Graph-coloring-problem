import pandas as pd
import numpy as np

def data_clean(path):
    df = pd.read_excel(path)
    df = df.sort_values(['session_code_n', 'round_number', 'id_in_group'])
    df = df.drop(df[df['actionType_n']==7].index)
    df = df.reset_index(drop=True)

    # merge the action type (one is timeout version)
    df.loc[df[df['actionType_n']==1].index, 'actionType_n'] = 0
    df.loc[df[df['actionType_n']==2].index, 'actionType_n'] = 0
    df.loc[df[df['actionType_n']==3].index, 'actionType_n'] = 1
    df.loc[df[df['actionType_n']==4].index, 'actionType_n'] = 1
    df.loc[df[df['actionType_n']==5].index, 'actionType_n'] = 2
    df.loc[df[df['actionType_n']==6].index, 'actionType_n'] = 2

    # change color index start from 0
    df.loc[df[df['new_color_n']==1].index, 'new_color_n'] = 0
    df.loc[df[df['new_color_n']==2].index, 'new_color_n'] = 1
    df.loc[df[df['new_color_n']==3].index, 'new_color_n'] = 2
    df.loc[df[df['new_color_n']==4].index, 'new_color_n'] = 3
    df.loc[df[df['color_n']==1].index, 'color_n'] = 0
    df.loc[df[df['color_n']==2].index, 'color_n'] = 1
    df.loc[df[df['color_n']==3].index, 'color_n'] = 2
    df.loc[df[df['color_n']==4].index, 'color_n'] = 3

    # add graph size colunm
    df['graph_size'] = 0
    session_num = 14
    session = [i for i in range(1, session_num)]
    for s in session:
        session_df = df[df['session_code_n'] == s]
        graph_size = len(session_df.drop_duplicates('id_in_group'))
        df['graph_size'].iloc[session_df.index] = graph_size

    # add history information
    df['hist_color'] = 0
    df['hist_neighbor'] = 0
    df['hist_skip'] = 0

    graph_size = list(df.drop_duplicates('session_code_n')['graph_size'])
    session_and_size_dict = {session[i]: graph_size[i] for i in range(len(session))}

    # 1 2: color, 3 4: skip, 5 6: player
    for s in session:
        session_df = df[df['session_code_n'] == s]
        user_hist = {i:[0, 0, 0] for i in range(1, session_and_size_dict[s]+1)}
        # print(len(user_hist))
        for idx, row in session_df.iterrows():
            id = row['id_in_group']
            round = row['round_number']-1
            action = row['actionType_n']
            if(round != 0):
                df.loc[idx, 'hist_color'] = np.round(user_hist[id][0] / round, 2)
                df.loc[idx, 'hist_skip'] = np.round(user_hist[id][1] / round, 2)
                df.loc[idx, 'hist_neighbor'] = np.round(user_hist[id][2] / round, 2)
            else:
                df.loc[idx, 'hist_color'] = np.round(user_hist[id][0], 2)
                df.loc[idx, 'hist_skip'] = np.round(user_hist[id][1], 2)
                df.loc[idx, 'hist_neighbor'] = np.round(user_hist[id][2], 2)


            if(action == 0):
                user_hist[id][0] += 1
            elif(action == 1):
                user_hist[id][1] += 1
            elif(action == 2):
                user_hist[id][2] += 1

    # add neughbor information
    df['num_of_neighbor'] = [len(str(neighbor_list).split(',')) for neighbor_list in df['linked_neighbors']]
    return df

def store_graph_info(df, path_df, path_structure, path_feature):
    # store graph feature and graph structure into different dataframe
    struct = {
        'session': df['session_code_n'], 
        'uid': df['id_in_group'], 
        'round': df['round_number'], 
        'linked': df['linked_neighbors']
    }
    graph_struct_df = pd.DataFrame(struct)

    feature = {
        'session': df['session_code_n'], 
        'uid': df['id_in_group'], 
        'round': df['round_number'], 
        'color': df['color_n'], 
        'score': df['score'], 
        'num_of_neighbor': df['num_of_neighbor'], 
        'hist_color': df['hist_color'], 
        'hist_neighbor': df['hist_neighbor'], 
        'hist_skip': df['hist_skip'], 
        'action': df['actionType_n']
    }
    graph_feature_df = pd.DataFrame(feature)

    df.to_csv(path_df, index=False)
    graph_struct_df.to_csv(path_structure, index=False)
    graph_feature_df.to_csv(path_feature, index=False)

def store_color_change_info(color_df, path):
    feature = {
        'session': color_df['session_code_n'], 
        'uid': color_df['id_in_group'], 
        'round': color_df['round_number'], 
        'color': color_df['color_n'], 
        'score': color_df['score'], 
        'num_of_neighbor': color_df['num_of_neighbor'], 
        'hist_color': color_df['hist_color'], 
        'hist_neighbor': color_df['hist_neighbor'], 
        'hist_skip': color_df['hist_skip'], 
        'new_color': color_df['new_color_n']
    }
    color_df = pd.DataFrame(feature)
    color_df.to_csv(path, index=False)

def store_neighbor_change_info(neighbor_df, path):
    feature = {
        'session': neighbor_df['session_code_n'], 
        'uid': neighbor_df['id_in_group'], 
        'round': neighbor_df['round_number'], 
        'color': neighbor_df['color_n'], 
        'score': neighbor_df['score'], 
        'num_of_neighbor': neighbor_df['num_of_neighbor'], 
        'hist_color': neighbor_df['hist_color'], 
        'hist_neighbor': neighbor_df['hist_neighbor'], 
        'hist_skip': neighbor_df['hist_skip'], 
        'del_neighbor': neighbor_df['delete_player_n'], 
        'add_neighbor': neighbor_df['add_player_n']
    }

    neighbor_df = pd.DataFrame(feature)
    neighbor_df.to_csv(path, index=False)


if __name__ == '__main__':
    path_input = 'Data/Color_Game_Endo.xlsx'
    path_output = 'Data/processed_data.csv'
    path_graph_feature = 'Data/graph_feature.csv'
    path_graph_structure = 'Data/graph_structure.csv'
    path_color_change_info = 'Data/change_color_df.csv'
    path_neighbor_change_info = 'Data/change_neighbor_df.csv'

    df = data_clean(path_input)
    store_graph_info(df, path_output, path_graph_structure, path_graph_feature)
    store_color_change_info(df, path_color_change_info)
    store_neighbor_change_info(df, path_neighbor_change_info)

    print('\ndata preprocessing done...\n')

