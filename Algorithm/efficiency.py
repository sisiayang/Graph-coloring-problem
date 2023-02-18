import numpy as np
def cal_node_score(G, s):
    score_list = []
    for color_set in s:
        for node in color_set:
            score = 6
            for i in G.graph.adj()[node].coalesce().indices()[0]:
                if(i in color_set):
                    score -= 1
            score_list.append(score)

    return np.round(np.mean(score_list), 2)