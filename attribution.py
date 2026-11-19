"""attribution.py — attribuzioni per lineare e MLP, e varianza predetta dalla Lyapunov."""
import numpy as np

def linear_attr(th, model, x):
    """φ_i = w_i x_i, con w = vista del blocco W1."""
    W = th[model.blocks[0][1]].reshape(model.blocks[0][2])
    return (W * x[:, None])  # (d, C)

def _fwd(model, th, x):
    P = [th[s].reshape(sh) for _, s, sh in model.blocks]
    Ws, bs = P[0::2], P[1::2]
    relu = getattr(model, "act", "relu") == "relu"
    a, Z = x, []
    for l in range(model.L):
        z = a @ Ws[l] + bs[l]
        Z.append(z)
        a = (np.maximum(z, 0) if relu else np.tanh(z)) if l < model.L - 1 else z
    return Ws, Z, a, relu

def input_grad(model, th, x, class_idx=None):
    """∂ logit_c / ∂x  (d,): gradiente PURO rispetto all'input. c = classe predetta in x se non specificata."""
    x = np.asarray(x, float).ravel()
    Ws, Z, a, relu = _fwd(model, th, x)
    J = Ws[-1]                                                   # (n_{L-1}, C)
    for l in range(model.L - 2, -1, -1):
        dact = (Z[l] > 0) if relu else 1.0 - np.tanh(Z[l]) ** 2
        J = (Ws[l] * dact[None, :]) @ J                          # (n_l, C)
    c = int(np.argmax(a)) if class_idx is None else int(class_idx)
    return J[:, c]

def grad_x_input(model, th, x, class_idx=None):
    """x ⊙ ∂ logit_c / ∂x. Restituisce (d,)."""
    x = np.asarray(x, float).ravel()
    return x * input_grad(model, th, x, class_idx)

def integrated_gradients(model, th, x, baseline=0.0, steps=50, class_idx=None):
    """IG con baseline costante. Classe FISSATA una volta in x; regola del punto medio; vale la completezza
    sum_i IG_i = f_c(x) - f_c(baseline) a meno dell'errore di quadratura."""
    x = np.asarray(x, float).ravel()
    b = np.full_like(x, baseline)
    c = int(np.argmax(_fwd(model, th, x)[2])) if class_idx is None else int(class_idx)
    al = (np.arange(steps) + 0.5) / steps
    g = np.mean([input_grad(model, th, b + a * (x - b), c) for a in al], axis=0)
    return (x - b) * g

def completeness_error(model, th, x, attr_vec, baseline=0.0, class_idx=None):
    """|sum(attr) - (f_c(x) - f_c(baseline))|: test di sanita' per IG."""
    x = np.asarray(x, float).ravel()
    a = _fwd(model, th, x)[2]
    c = int(np.argmax(a)) if class_idx is None else int(class_idx)
    f0 = _fwd(model, th, np.full_like(x, baseline))[2][c]
    return abs(float(np.sum(attr_vec)) - float(a[c] - f0))

def get_attr(cfg):
    """Restituisce attr(model, th, x, class_idx) secondo cfg.attr_method (grad_x_input | ig)."""
    if cfg.attr_method == "ig":
        return lambda m, th, x, c=None: integrated_gradients(m, th, x, steps=cfg.ig_steps, class_idx=c)
    return grad_x_input

def predicted_cov(J_phi, U, S):
    """Cov(φ) ≈ J_phi U S U^T J_phi^T, con J_phi (n_attr, N), U (N,k), S (k,k)."""
    M = J_phi @ U
    return M @ S @ M.T
