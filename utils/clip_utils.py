import torch
# float[H, W, C], float[P, C], float[N, C] -> float[H, W, ]
def get_relevancy(raw_semantic_map: torch.Tensor, pembed: torch.Tensor, nembed: torch.Tensor):
    s = raw_semantic_map.shape[:-1]
    c = raw_semantic_map.shape[-1]
    raw_semantics = raw_semantic_map.flatten(0, -2)
    psim=pembed@raw_semantics.T # (p, i)
    nsim=nembed@raw_semantics.T # (n, i)
    nsim=nsim.unsqueeze(0).repeat_interleave(pembed.shape[0],dim=0) # (p, n ,i)
    psim=psim.unsqueeze(1).repeat_interleave(nembed.shape[0],dim=1) # (p, n, i)
    sim=torch.stack((psim,nsim), dim=-1) # (p, n, i, 2)
    sim=torch.softmax(10*sim, dim=-1) # (p, n, i, 2)
    sim, indice = sim[...,0].min(dim=1) # (p, i)
    return sim.unflatten(1, s)