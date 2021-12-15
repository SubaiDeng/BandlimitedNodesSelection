import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_graph():
    num_nodes = 12
    nodes = list(range(num_nodes))
    edges = [(1, 2),
             (1, 3),
             (2, 3),
             (2, 4),
             (4, 5),
             (4, 9),
             (5, 6),
             (5, 8),
             (6, 7),
             (7, 8),
             (9, 10),
             (10, 11),
             (11, 0),
             (0, 9)]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # nx.draw(g)
    # plt.show()
    return g


def greedy_estimating(g, L, M, k):
    L_trans = np.transpose(L)
    L_k = L
    L_trans_k = L_trans
    for i in range(k-1):
        L_trans_k = np.matmul(L_trans_k, L_trans)
    for i in range(k-1):
        L_k = np.matmul(L_k, L)
    L_merge_k = np.matmul(L_trans_k, L_k)
    S = list()# sampling node set

    while len(S) < M:
        S_c = np.eye(len(L))
        for i in S:
            S_c[i][i] = 0
        L_pi = np.matmul(np.matmul(S_c, L_merge_k), S_c)
        w, v = np.linalg.eigh(L_pi)
        min_ev_idx = np.argmin(w)
        v = v[:, min_ev_idx]
        v_idx = np.argmax(v)
        S.append(v_idx)
    S_opt = S
    return S_opt


def print_result(g, L, M, k, S_opt):
    print('GRAPH CONSTRUCTION')
    print('nodes:')
    print(g.nodes)
    print('edges')
    print(g.edges)
    print('Parameter:')
    print('M:{}'.format(M))
    print('k:{}'.format(k))
    print('Selection List:')
    print(S_opt)


def main():
    print("START")


    # Create Graph
    g = create_graph()
    # Laplacian : using symmetric normalized Random walk version
    A = nx.to_numpy_matrix(g).A
    deg = np.array(np.sum(A, axis=0))

    num_nodes = len(deg)
    I = np.eye(num_nodes)
    deg_inv_half = np.power(deg, -1/2)
    # t = np.matmul(np.matmul(np.diag(deg_inv_half), A), np.diag(deg_inv_half))
    L = I - np.matmul(np.matmul(np.diag(deg_inv_half), A), np.diag(deg_inv_half))

    M = 4
    k = 10
    S_opt = greedy_estimating(g, L, M, k)

    print_result(g, L, M, k, S_opt)


if __name__ == '__main__':
    main()

