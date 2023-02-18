import matplotlib.pyplot as plt
import networkx as nx


def show_solution(solution, num_of_steps):
    if(solution):
        print('The number of rounds taken to solve the problem: ', num_of_steps)
        for i in range(len(solution)):
            print(f'color {i+1}: {sorted(solution[i])}')
    else:
        print('solution doesn\'t exist!')


def show_graph(nx_g, solution=None):
    c = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', '#9467bd']
    plt.figure(figsize=[5,5])
    pos = nx.circular_layout(nx_g)  # 環狀布圖

    if(solution == None):
        nx.draw_networkx_nodes(nx_g, pos, node_color='tab:blue')
        nx.draw_networkx_edges(nx_g, pos, alpha=0.5, width=1)

    else:
        idx_to_node = {}
        idx = 0
        for n in sorted(list(nx_g.nodes)):
            idx_to_node[idx] = n
            idx += 1

        new_solution = []
        print(solution)
        for s in range(len(solution)):
            new_s = []
            for idx in solution[s]:
                new_s.append(idx_to_node[idx])
            new_solution.append(new_s)
        for i in range(len(new_solution)):
            nx.draw_networkx_nodes(nx_g, pos, nodelist=new_solution[i], node_color=c[i])
        nx.draw_networkx_edges(nx_g, pos, alpha=0.5, width=1)