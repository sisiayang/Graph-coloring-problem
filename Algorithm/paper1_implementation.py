import dgl
import random as rd
import copy
from create_graph import make_ring_lattice
from efficiency import cal_node_score
import numpy as np
from tqdm import tqdm

class Graph(object):
    def __init__(self, size, neighber):
        self.nx_g = make_ring_lattice(size, neighber)
        self.graph = dgl.from_networkx(self.nx_g)
        self.nodes = [int(i) for i in list(self.graph.nodes())]
        self.edges = [(int(self.graph.edges()[0][i]), int(self.graph.edges()[1][i])) for i in range(len(self.graph.edges()[0]))]
        self.size = size
        self.best_solution = []
        self.best_conflict = size  # init = 最大的可能conflict數量
        self.success_round = 0

def initialize(G, k):
    graph = G.graph
    color_class = [[] for i in range(k)]    # k 個color set
    inserted = []
    for node in G.nodes:
        for c in range(len(color_class)):  # each class
            allowed_class = 1   # 是否允許加入此class
            for i in color_class[c]:    # each node in this class
                if(graph.has_edges_between(node, i)):   # 存在共邊
                    allowed_class = 0
                    break
            if(allowed_class == 1):
                color_class[c].append(node)
                inserted.append(node)
                break
    for node in G.nodes:
        if(node not in inserted):   # 尚未分配class -> each color都有conflict
            # random加入
            idx = rd.randint(0, k-1)
            color_class[idx].append(node)

    return color_class

def fitness_fn(G, s):   # 目標是最小化output -> 0代表沒有衝突
    edges = G.edges
    output = 0
    # check each edge的兩點是否在不同color set
    for edge in edges:
        for color_class in s:
            if(edge[0] in color_class and edge[1] in color_class):
                output += 1
                break
    return output

def choose_P(P):    # random choose two parents
    while(True):
        num1 = rd.randint(0, len(P)-1)
        num2 = rd.randint(0, len(P)-1)
        if(num1 != num2):
            return P[num1], P[num2]

def remove_nodes(s, nodes):
    # use in crossover to create new solution
    for n in nodes:
        for node_set in s:
            if(n in node_set):
                node_set.remove(n)
                break

def crossover(s1, s2, k):
    s = [[] for i in range(k)]
    tmp_s = [copy.deepcopy(s1), copy.deepcopy(s2)]
    s_idx = -1
    for i in range(k):
        s_idx = i % 2 # odd: s2, even: s1
        
        # find the max length set
        max_set = []
        for set in tmp_s[s_idx]:
            if(len(set) > len(max_set)):
                max_set = copy.deepcopy(set)
        s[i] = copy.deepcopy(max_set)
        remove_nodes(tmp_s[0], max_set)
        remove_nodes(tmp_s[1], max_set)
        
    # 剩下的random加入
    for set in tmp_s[0]:
        for n in set:
            idx = rd.randint(0, k-1)
            s[idx].append(n)
    return s

def find_conflict_node(edges, tabulist, s):
    for edge in edges:
        # G[0][i] & G[1][i] 為一條邊的兩端
        for i in range(len(s)):
            if(edge[0] in s[i] and edge[1] in s[i]):
                move_node = edge[0] # edge[0]和edge[1]互相衝突 (都在i這個color set裡面)

                if(0 in tabulist[move_node]):   # tabulist中至少有一個class可以move
                    move_set = i
                    while(move_set == i or tabulist[move_node][move_set] != 0):
                        move_set = rd.randint(0, len(s)-1)
                    return move_node, i, move_set
                
                else:
                    break   # 下一條edge

def localSearch(G, s, L):
    nodes = G.nodes
    edges = G.edges
    tabulist = [[0 for i in range(len(s))] for j in range(len(nodes))]
    nb_cfl = fitness_fn(G, s)
    A = 3
    alpha = 0.6
    
    for l in range(L):
        # tabulist所有非0元素-1
        for row in range(len(tabulist)):
            for col in range(len(tabulist[0])):
                tabulist[row][col] = max(0, tabulist[row][col]-1)

        if(fitness_fn(G, s) == 0):
            return s
        # s has some conflict node
        move_node, color_class, move_set = find_conflict_node(edges, tabulist, s)
        
        # move & update tabulist
        s[color_class].remove(move_node)
        s[move_set].append(move_node)
        tabulist[move_node][color_class] = rd.randint(0, A) + alpha*nb_cfl
    return s

def updatePopulation(P, G, s, s1, s2):
    if(fitness_fn(G, s1) < fitness_fn(G, s2)):
        P.remove(s2)
    else:
        P.remove(s1)
    P.append(s)
    return P

def stopCondition(G, P):
    for p in P:
        # 至少有一組全域解 in P
        if(not fitness_fn(G, p)):
            return True
    return False

def run(G):
    L = 3
    k = 4
    P = []
    p = 5  # num of the Parent
    stop_steps = 15
    user_score = []

    # initialize p個solution
    for i in range(p):
        p = initialize(G, k)
        p = localSearch(G, p, L)
        if(not fitness_fn(G, p)):
            return p, 0, 0, [6]
        P.append(p)

    while(not stopCondition(G, P) and stop_steps):
        s1, s2 = choose_P(P)
        s = crossover(s1, s2, k)
        s = localSearch(G, s, L)
        user_score.append(cal_node_score(G, s)) # 針對新的solution s計算user score
        P = updatePopulation(P, G, s, s1, s2)
        stop_steps -= 1

    # 從P中找出最佳解作為final solution
    best_solution = P[0]
    best_conflict = len(G.nodes)
    for p in P:
        conflict = fitness_fn(G, p)
        if(not conflict):   # conflict = 0
            num_of_steps = 50-stop_steps
            return p, conflict, num_of_steps, user_score
        else:
            if(conflict < best_conflict):
                best_solution = p
                best_conflict = conflict
    return best_solution, best_conflict, 50, user_score

def avg_score_for_each_round(user_score_hist):
    score_for_each_round = np.mean(user_score_hist, axis=0).tolist()
    avg_score = [np.round(i, 2) for i in score_for_each_round]
    
    std = np.std(user_score_hist, axis=0).tolist()
    std = [np.round(i, 2) for i in std]

    return avg_score, std

if __name__ == '__main__':
    graph_size = [15, 16, 17, 18, 20, 21, 22, 23, 25, 26]
    record_path = 'Result/record_paper1.txt'
    f = open(record_path, 'w')
    trails = 10

    for size in tqdm(graph_size):
        user_score_hist = []
        G = Graph(size, 6)
        conflict_hist = []
        for iter in range(trails):
            solution, conflict, num_of_steps, user_score = run(G)

            user_score_hist.append(user_score)
            conflict_hist.append(conflict)
            if(conflict == 0):
                G.success_round += 1

            if(conflict < G.best_conflict):
                G.best_solution = solution
                G.best_conflict = conflict
        avg_score, std = avg_score_for_each_round(user_score_hist)

        f.writelines(['graph size: ', str(size), '\n'])
        f.writelines(['mean conflict: ', str(np.mean(conflict_hist)), '\n'])
        f.writelines(['best conflict: ', str(G.best_conflict), ', best solution: ', str(G.best_solution), '\n\n'])
        f.writelines(['avg score for each round: ', str(avg_score[:15]), '\n'])
        f.writelines(['std for each round: ', str(std), '\n'])
        f.writelines(['percentage of success round: ', str(np.round(G.success_round/trails, 2))])
        f.writelines(['\n', '='*80, '\n\n'])

    f.close()