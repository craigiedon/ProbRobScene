from scipy.optimize import linprog
import numpy as np


def multi_max_flow(sources, terminals, weight_matrix):
    super_source_row = [np.inf if i in sources else 0.0 for i in range(len(weight_matrix))]
    super_terminal_col = [np.inf if i in terminals else 0.0 for i in range(len(weight_matrix))]

    nodes = len(weight_matrix)

    aug_wm = np.zeros((nodes + 2, nodes + 2))
    aug_wm[0, 2:] = super_source_row
    aug_wm[2:, 1] = super_terminal_col
    aug_wm[2:, 2:] = weight_matrix

    return max_flow(0, 1, aug_wm)


def max_flow(source, terminal, weight_matrix):
    edges = []
    for u in range(len(weight_matrix)):
        for v in range(len(weight_matrix[0])):
            if u == v:
                weight_matrix[u][v] = 0 # eliminate self loops
            weight = weight_matrix[u][v]
            if weight != 0:
                edges.append((u, v))

    objective_coefficients = np.array([1 if u == source else 0 for (u, v) in edges])
    bounds = [(0, weight_matrix[u][v]) for u, v in edges]

    flow_constraints = []
    for node in range(0, len(weight_matrix)):
        if node == source or node == terminal:
            continue
        flow_constraint = np.zeros(len(edges))
        for i, (from_node, to_node) in enumerate(edges):
            if from_node == node:
                flow_constraint[i] = -1
            elif to_node == node:
                flow_constraint[i] = 1
        flow_constraints.append(flow_constraint)

    result = linprog(-objective_coefficients, A_eq=flow_constraints, b_eq=np.zeros(len(flow_constraints)), bounds=bounds, method='revised simplex')

    m_flow = int(np.round(-result.fun))

    return m_flow


print(multi_max_flow([0], [3], [
    [0, 7, 0, 0],
    [0, 0, 6, 0],
    [0, 0, 0, 8],
    [9, 0, 0, 0]
]))

print(multi_max_flow([0, 1], [4, 5], [[0, 0, 4, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))

print(multi_max_flow([0], [1], [[1, 1], [1, 0]]))