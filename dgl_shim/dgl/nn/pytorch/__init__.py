from torch_geometric.utils import softmax as _softmax


def edge_softmax(graph, logits):
    """Softmax of edge logits over the incoming edges of each destination node (DGL edge_softmax)."""
    if logits.shape[0] == 0:
        return logits
    return _softmax(logits, graph.dst, num_nodes=graph.number_of_nodes())
