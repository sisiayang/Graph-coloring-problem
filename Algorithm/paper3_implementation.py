import dgl
import random as rd
import numpy as np
import math
import copy
from collections import deque
from create_graph import make_ring_lattice
from efficiency import cal_node_score
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

def fitness_fn(G, s):   # 目標是最小化output -> 0代表沒有衝突
    edges = G.edges
    output = 0
    for edge in edges:
        for color_class in s:
            if(edge[0] in color_class and edge[1] in color_class):
                output += 1
                break
    return output

def is_conflict(G, color_set, node):
    graph = G.graph
    for n in color_set:
        if(graph.has_edges_between(node, n)):
            return True
    return False

def find_new_class(G, s, node, idx):
    graph = G.graph
    for color_idx in range(len(s)):
        allowed = True
        for i in s[color_idx]:
            if(graph.has_edges_between(node, i)):
                allowed = False
                break
        if(allowed):
            return color_idx
    # 都沒有
    color_idx = rd.randint(0, len(s)-1)
    while(color_idx == idx):
        color_idx = rd.randint(0, len(s)-1)
    return color_idx

def generate_neighbor(G, s, k):
    neighborhood_set = []
    for color_idx in range(k):
        for node in s[color_idx]:
            if(is_conflict(G, s[color_idx], node)):
                new_s = copy.deepcopy(s)
                new_s[color_idx].remove(node)
                new_s[find_new_class(G, new_s, node, color_idx)].append(node)
                neighborhood_set.append(new_s)
    return neighborhood_set

def SA(G, s, k, UnAc, UnIm):
    T = 1
    SF = 1  # user define (p.15)
    TDR = 0.85  # 0.8 ~ 0.99
    IterCount = len(G.nodes)*k*SF
    
    unaccept = 0
    unimprove = 0
    while(True):
        for iter in range(IterCount):
            # stop condition -> UnAc or UnIm or find solution
            if(unaccept==UnAc or not fitness_fn(G, s) or unimprove==UnIm):
                return s, unimprove, unaccept

            # choose多組可能的solution
            neighborhood_set = generate_neighbor(G, s, k)

            # 從多組解中random選一個
            new_s = neighborhood_set[rd.randint(0, len(neighborhood_set)-1)]

            # 是否improve
            delta = fitness_fn(G, new_s) - fitness_fn(G, s)
            if(delta < 0):  # improve -> change
                unimprove = 0
                s = new_s
            else:   # probability choose
                unimprove += 1
                prob = math.exp(-delta/T)
                if(rd.choices([0, 1], weights=[1-prob, prob])[0]):
                    s = new_s
                else:   # unaccess
                    unaccept += 1
        T = TDR * T

def tabu_candidate(G, s, k):
    neighborhood_set = []
    for color_idx in range(k):
        for node in s[color_idx]:
            if(is_conflict(G, s[color_idx], node)):
                new_s = copy.deepcopy(s)

                new_class = find_new_class(G, new_s, node, color_idx)
                new_s[color_idx].remove(node)
                new_s[new_class].append(node)

                neighborhood_set.append(((node, new_class, color_idx), fitness_fn(G, new_s)))
    return neighborhood_set

def update_tabulist(tabulist):
    for raw in tabulist:
        for col in raw:
            col = max(0, col-1)

def TS(G, s, k):
    tabulist = [[0 for i in range(k)] for j in range(len(G.nodes))]
    TSL = 50
    A = 15
    alpha = 0.6
    while(not fitness_fn(G, s) and TSL):
        # return (<v, i>, conflictNum) pair
        neighborhood_set = tabu_candidate(G, s, k)
        # choose one 符合資格的s -> 符合tabulist規範的best solution (根據fitness_fn排序)
        new_s = []
        conflict = 0
        neighborhood_set = sorted(neighborhood_set, key=lambda s: s[1])
        for candidate_s, conflictNum in neighborhood_set:
            if(tabulist[candidate_s[0]][candidate_s[1]] == 0):
                new_s = copy.deepcopy(s)
                new_s[candidate_s[2]].remove(candidate_s[0])
                new_s[candidate_s[1]].append(candidate_s[1])
                tabulist[candidate_s[0]][candidate_s[1]] = int(rd.randint(0, len(A))+alpha*conflictNum)
                conflict = conflictNum
                break
        
        # 更新tabulist
        update_tabulist(tabulist)

        # if improve -> 換
        if(conflict < fitness_fn(G, s)):
            s = new_s
        TSL -= 1
    return s

def find_next(G, visited, node):
    graph = G.graph
    neighbor = list(graph.in_edges(node))[0]  # neighbors of node i
    neighbor_degree = [int(graph.in_degrees(i)) for i in list(graph.in_edges(node))[0]]
    sorted_neighbor = sorted([(int(neighbor[i]), neighbor_degree[i]) for i in range(len(neighbor))], key=lambda n:n[1], reverse=True)
    for n in sorted_neighbor:
        if(n[0] not in visited):
            return n[0]
    return None

def find_color(G, s, node):
    graph = G.graph
    for color_class in range(len(s)):
        allowed_class = 1
        for n in s[color_class]:
            if(graph.has_edges_between(n, node)):
                allowed_class = 0
                break
        if(allowed_class == 1):
            return color_class
    if(allowed_class == 0):
        return rd.randint(0, len(s)-1)

def Initialize(G, k):
    graph = G.graph
    init_s = [[] for i in range(k)]
    visited = []
    stack = deque()

    # 最大degree的node先放入
    max_degree_node = int(np.argmax(graph.in_degrees()))
    init_s[0].append(max_degree_node)
    visited.append(max_degree_node)
    stack.append(max_degree_node)
    node = find_next(G, visited, max_degree_node)
    while(stack):
        init_s[find_color(G, init_s, node)].append(node)
        # 把現在的node丟進stack
        stack.append(node)
        visited.append(node)
        # 找下一個node
        node = find_next(G, visited, node)
        while(not node and stack):
            node = find_next(G, visited, stack[-1])
            if(not node):
                stack.pop()
    return init_s

def run(G):
    user_score = []   # 記錄每輪每個node的score (針對每輪新的s)
    k = 4
    s = Initialize(G, k)
    maxIter = 15
    UnIm = 5
    UnAc = 20

    if(not fitness_fn(G, s)):
        user_score = [6]

    # if 在一輪中SA和TS都找不到比較好的解 -> 直接結束
    while(maxIter and fitness_fn(G, s)):
        new_s, unimprove, unaccept = SA(G, s, k, 20, 5)
        if(unimprove > UnIm or unaccept > UnAc):
            new_s = TS(G, new_s, k)
        
        user_score.append(cal_node_score(G, new_s)) # 針對新的solution s計算user score

        # if(new_s == s):
        #     break
        # else:
        #     s = new_s
        s = new_s
        maxIter -= 1

    conflict = fitness_fn(G, s)
    return s, conflict, 15-maxIter, user_score

def avg_score_for_each_round(user_score_hist):
    score_for_each_round = np.mean(user_score_hist, axis=0).tolist()
    avg_score = [np.round(i, 2) for i in score_for_each_round]
    
    std = np.std(user_score_hist, axis=0).tolist()
    std = [np.round(i, 2) for i in std]

    return avg_score, std

if __name__ == '__main__':
    graph_size = [15, 16, 17, 18, 20, 21, 22, 23, 25, 26]
    record_path = 'Result/record_paper3.txt'
    f = open(record_path, 'w')
    trails = 10

    for size in tqdm(graph_size):
        G = Graph(size, 6)
        user_score_hist = []
        conflict_hist = []
        for iter in range(trails):
            
            solution, conflict, num_of_step, user_score = run(G)

            user_score_hist.append(user_score)
            conflict_hist.append(conflict)
            if(conflict == 0):
                G.success_round += 1

            if(conflict < G.best_conflict):
                G.best_solution = solution
                G.best_conflict = conflict
        avg_score, std = avg_score_for_each_round(user_score_hist)

        f.writelines(['graph size: ', str(size), '\n'])
        f.writelines(['mean conflict: ', str(np.round(np.mean(conflict_hist), 2)), '\n'])
        f.writelines(['best conflict: ', str(G.best_conflict), ', best solution: ', str(G.best_solution), '\n\n'])
        f.writelines(['avg score for each round: ', str(avg_score[:15]), '\n'])
        f.writelines(['std for each round: ', str(std), '\n'])
        f.writelines(['percentage of success round: ', str(np.round(G.success_round/trails, 2))])
        f.writelines(['\n', '='*80, '\n\n'])

    f.close()