"""data.py — dati sintetici gaussiani con spettro controllato, e loader per digits/MNIST/CIFAR10 da data/."""
import os
import numpy as np

def make_synthetic(cfg):
    rng = np.random.default_rng(cfg.seed_data)
    d, C = cfg.d, cfg.n_classes
    # rotazione casuale e spettro a coda lunga
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Cmat = Q @ np.diag(cfg.spectrum[:d]) @ Q.T
    w_star = rng.normal(size=(d, C))
    n = cfg.n_train + cfg.n_test
    X = rng.normal(size=(n, d)) @ np.linalg.cholesky(Cmat).T
    if cfg.teacher == "mlp":                      # insegnante non lineare: la MLP non e' piu' quasi-lineare
        A, B = rng.normal(size=(d, 16)) / np.sqrt(d), rng.normal(size=(16, C))
        logits = np.tanh(X @ A) @ B
        logits *= 2.0 / logits.std()
    else:
        logits = X @ w_star
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > np.cumsum(prob, 1)).sum(1).clip(0, C - 1)   # campionamento categorico vettoriale
    return dict(Xtr=X[:cfg.n_train].astype(np.float64), ytr=y[:cfg.n_train],
                Xte=X[cfg.n_train:].astype(np.float64), yte=y[cfg.n_train:],
                C=Cmat, w_star=w_star, active_pix=np.ones(d, bool))

def make_synthetic_reg(cfg):
    """Regressione lineare gaussiana, y = w*.x + sigma*eps: il modello esatto della derivazione (task='reg')."""
    rng = np.random.default_rng(cfg.seed_data + 1)
    d = cfg.d
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Cmat = Q @ np.diag(cfg.spectrum[:d]) @ Q.T
    w_star = rng.normal(size=d)
    n = cfg.n_train + cfg.n_test
    X = rng.normal(size=(n, d)) @ np.linalg.cholesky(Cmat).T
    y = X @ w_star + cfg.sigma * rng.normal(size=n)
    return dict(Xtr=X[:cfg.n_train], ytr=y[:cfg.n_train], Xte=X[cfg.n_train:], yte=y[cfg.n_train:],
                C=Cmat, w_star=w_star, sigma=cfg.sigma, active_pix=np.ones(d, bool), task="reg")

def _load_npz_or_raw(name, cfg):
    """Cerca data/<name>.npz, altrimenti usa sklearn/torchvision se presenti."""
    p = os.path.join(cfg.data_dir, f"{name}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data.astype(np.float64), d.target
        ntr = int(0.75 * len(X))
        return X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    if name == "mnist":
        from torchvision.datasets import MNIST
        tr, te = MNIST(cfg.data_dir, train=True, download=True), MNIST(cfg.data_dir, train=False, download=True)
        return (tr.data.numpy().reshape(len(tr), -1) / 255.0).astype(np.float64), tr.targets.numpy(), \
               (te.data.numpy().reshape(len(te), -1) / 255.0).astype(np.float64), te.targets.numpy()
    if name == "cifar10":
        from torchvision.datasets import CIFAR10
        tr, te = CIFAR10(cfg.data_dir, train=True, download=True), CIFAR10(cfg.data_dir, train=False, download=True)
        f = lambda ds: (ds.data.reshape(len(ds), -1) / 255.0).astype(np.float64)
        return f(tr), np.array(tr.targets), f(te), np.array(te.targets)
    raise FileNotFoundError(f"dataset {name} non trovato in {cfg.data_dir}")

def load(cfg):
    if cfg.dataset == "synthetic":
        out = make_synthetic(cfg)
        out["reg"] = make_synthetic_reg(cfg)          # dati di regressione per C0/C3 (modello lineare esatto)
        return out
    Xtr, ytr, Xte, yte = _load_npz_or_raw(cfg.dataset, cfg)
    cfg.d = Xtr.shape[1]
    cfg.n_classes = int(max(ytr.max(), yte.max()) + 1)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, active_pix=Xtr.max(0) > 0)