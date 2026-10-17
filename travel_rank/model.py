"""Small CPU LTR models using official PyTorch APIs; not a paper reproduction."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def make_model(features=6, hidden=16, seed=7):
    """O(FH) time/space; one hidden layer."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def ranking_loss(scores, grades, method):
    """Point/list O(BM); pair O(BM²) time/space. Fixed-size, fully judged lists."""
    if scores.shape != grades.shape or scores.ndim != 2:
        raise ValueError("scores/grades must have matching [batch, candidates] shapes")
    if not torch.isfinite(scores).all() or not torch.isfinite(grades).all() or (grades < 0).any() or (grades > 4).any():
        raise ValueError("finite scores and grades in [0,4] required")
    if method == "pointwise":
        return F.binary_cross_entropy_with_logits(scores, (grades > 0).float())
    if method == "pairwise":
        mask = grades.unsqueeze(2) > grades.unsqueeze(1)
        differences = scores.unsqueeze(2) - scores.unsqueeze(1)
        losses = F.softplus(-differences) * mask
        counts = mask.sum((1, 2))
        active = counts > 0
        per_query = losses.sum((1, 2)) / counts.clamp_min(1)
        return per_query[active].mean() if active.any() else scores.sum() * 0
    if method == "listwise":
        gains = torch.pow(2.0, grades) - 1
        totals = gains.sum(1, keepdim=True)
        active = totals.squeeze(1) > 0
        targets = gains / totals.clamp_min(1)
        per_query = -(targets * F.log_softmax(scores, dim=1)).sum(1)
        return per_query[active].mean() if active.any() else scores.sum() * 0
    raise ValueError("unknown loss")


def tensors(rows):
    """O(QMF) time/space. Materializes the bounded synthetic dataset on CPU."""
    if not rows:
        raise ValueError("empty partition")
    return torch.tensor([r["x"] for r in rows], dtype=torch.float32), torch.tensor([r["y"] for r in rows], dtype=torch.float32)


def fit(train, valid, method="pairwise", epochs=40, patience=6, batch_size=32, seed=7):
    """Bounded mini-batch training; see complexity.md for full epoch cost."""
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("positive training settings required")
    x, y = tensors(train)
    vx, vy = tensors(valid)
    model = make_model(x.shape[-1], seed=seed)
    mean = x.mean((0, 1))
    scale = x.std((0, 1), unbiased=False).clamp_min(1e-6)
    x, vx = (x - mean) / scale, (vx - mean) / scale
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for indices in order.split(batch_size):
            opt.zero_grad(set_to_none=True)
            loss = ranking_loss(model(x[indices]).squeeze(-1), y[indices], method)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.inference_mode():
            value = float(ranking_loss(model(vx).squeeze(-1), vy, method))
        history.append(value)
        if value < best - 1e-6:
            best, best_state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, scale, history


@torch.inference_mode()
def predict(model, mean, scale, features):
    """O(QMFH) time; O(QMH+QMF) space. Input size bounded by caller."""
    model.eval()
    x = torch.tensor(features, dtype=torch.float32)
    return model((x - mean) / scale).squeeze(-1).tolist()
