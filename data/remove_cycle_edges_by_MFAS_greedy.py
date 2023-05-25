"""
Taken from https://github.com/zhenv5/breaking_cycles_in_noisy_hierarchies
"""
import networkx as nx
from s_c_c import filter_big_scc, get_big_sccs, scc_nodes_edges


def get_nodes_degree_dict(g, nodes):
    in_degrees = g.in_degree(nodes)
    out_degrees = g.out_degree(nodes)
    degree_dict = {}
    for node in nodes:
        in_d = in_degrees[node]
        out_d = out_degrees[node]
        if in_d >= out_d:
            try:
                value = in_d * 1.0 / out_d
            except Exception:
                value = 0
            f = "in"
        else:
            try:
                value = out_d * 1.0 / in_d
            except Exception:
                value = 0
            f = "out"
        degree_dict[node] = (value, f)
        # print("node: %d: %s" % (node,degree_dict[node]))
    return degree_dict


def greedy_local_heuristic(sccs, degree_dict, edges_to_be_removed):
    while True:
        graph = sccs.pop()
        temp_nodes_degree_dict = {}
        for node in graph.nodes():
            temp_nodes_degree_dict[node] = degree_dict[node][0]
        from helper_funs import pick_from_dict

        max_node, _ = pick_from_dict(temp_nodes_degree_dict)
        max_value = degree_dict[max_node]
        # degrees = [(node,degree_dict[node]) for node in list(graph.nodes())]
        # max_node,max_value = max(degrees,key = lambda x: x[1][0])
        if max_value[1] == "in":
            # indegree > outdegree, remove out-edges
            edges = [(max_node, o) for o in graph.neighbors(max_node)]
        else:
            # outdegree > indegree, remove in-edges
            edges = [(i, max_node) for i in graph.predecessors(max_node)]
        edges_to_be_removed += edges
        sub_graphs = filter_big_scc(graph, edges_to_be_removed)
        if sub_graphs:
            for index, sub in enumerate(sub_graphs):
                sccs.append(sub)
        if not sccs:
            return


def remove_cycle_edges_by_mfas(graph):
    from remove_self_loops import remove_self_loops_from_graph

    self_loops = remove_self_loops_from_graph(graph)

    scc_nodes, _, _, _ = scc_nodes_edges(graph)
    degree_dict = get_nodes_degree_dict(graph, scc_nodes)
    sccs = get_big_sccs(graph)
    if len(sccs) == 0:
        print(
            "After removal of self loop edges: %s" % nx.is_directed_acyclic_graph(graph)
        )
        return self_loops
    edges_to_be_removed = []
    import timeit

    t1 = timeit.default_timer()
    greedy_local_heuristic(sccs, degree_dict, edges_to_be_removed)
    t2 = timeit.default_timer()
    print("mfas time usage: %0.4f s" % (t2 - t1))
    edges_to_be_removed = list(set(edges_to_be_removed))
    edges_to_be_removed += self_loops
    return edges_to_be_removed
