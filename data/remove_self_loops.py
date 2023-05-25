"""
Taken from https://github.com/zhenv5/breaking_cycles_in_noisy_hierarchies
"""
import networkx as nx


def remove_self_loops_from_graph(g):
    self_loops = list(nx.selfloop_edges(g))
    g.remove_edges_from(self_loops)
    return self_loops
