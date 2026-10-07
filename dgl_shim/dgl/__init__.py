"""Minimal stand-in for the few DGL features used by the official HGNN-AC/MAGNN code
(DGLGraph construction, edge data, update_all with a sum reducer, edge_softmax).
DGL has no build for this PyTorch/CUDA (RTX 5090), so these are re-implemented with plain torch.
Semantics follow DGL 0.4–0.6: edges keep insertion order; messages flow src -> dst; sum over in-edges."""
import torch
from . import function  # noqa: F401


class _EdgeBatch:
    def __init__(self, data):
        self.data = data


class DGLGraph:
    def __init__(self, multigraph=True):
        self._n = 0
        self.src = torch.empty(0, dtype=torch.long)
        self.dst = torch.empty(0, dtype=torch.long)
        self.edata, self.ndata = {}, {}
        self.device = torch.device("cpu")

    def add_nodes(self, n):
        self._n += int(n)

    def add_edges(self, u, v):
        u = torch.as_tensor(list(u), dtype=torch.long)
        v = torch.as_tensor(list(v), dtype=torch.long)
        self.src = torch.cat([self.src, u.to(self.src.device)])
        self.dst = torch.cat([self.dst, v.to(self.dst.device)])

    def number_of_nodes(self):
        return self._n

    def number_of_edges(self):
        return int(self.src.numel())

    def to(self, device):
        self.device = torch.device(device)
        self.src, self.dst = self.src.to(self.device), self.dst.to(self.device)
        return self

    def update_all(self, message_func, reduce_func):
        op, msg_field, out_field = reduce_func
        assert op == "sum"
        msg = message_func(_EdgeBatch(self.edata))[msg_field]
        out = torch.zeros((self._n,) + tuple(msg.shape[1:]), dtype=msg.dtype, device=msg.device)
        if msg.shape[0] > 0:
            out.index_add_(0, self.dst, msg)
        self.ndata[out_field] = out
