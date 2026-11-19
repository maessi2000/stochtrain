"""models.py — LinearModel, LinearReg (MSE) e MLP con interfaccia identica. Curvature (Hessiana via Hv).

CONVENZIONE: loss_grad restituisce il gradiente dei SOLI dati. Il weight decay si applica una volta sola, in sgd();
l'argomento wd dei costruttori e' mantenuto per compatibilita' ma non viene usato."""
import numpy as np

class _Base:
    """Solo i metadati comuni; N, blocks, layer_of, post_of, pre_of, is_weight."""
    def _build_meta(self, sizes):
        self.sizes, self.L = tuple(sizes), len(sizes) - 1
        self.blocks, off = [], 0
        for l in range(self.L):
            ni, no = sizes[l], sizes[l + 1]
            self.blocks.append((f"W{l+1}", slice(off, off + ni * no), (ni, no))); off += ni * no
            self.blocks.append((f"b{l+1}", slice(off, off + no), (no,))); off += no
        self.N = off
        self.names = [b[0] for b in self.blocks]
        self.layer_of = np.zeros(off, np.int8)
        self.post_of = np.zeros(off, np.int32)
        self.pre_of = -np.ones(off, np.int32)
        for l in range(self.L):
            (_, sW, (ni, no)), (_, sb, _) = self.blocks[2*l], self.blocks[2*l+1]
            self.layer_of[sW] = self.layer_of[sb] = l
            p = np.arange(ni * no)
            self.post_of[sW] = sum(sizes[:l+1]) + p % no
            self.post_of[sb] = sum(sizes[:l+1]) + np.arange(no)
            self.pre_of[sW] = sum(sizes[:l]) + p // no
        self.is_weight = np.array([n.startswith("W") for n in self.names])

    def dead_params(self, active_pix):
        pre = self.pre_of
        return (pre >= 0) & (pre < self.sizes[0]) & ~active_pix[np.clip(pre, 0, self.sizes[0]-1)]

class LinearModel(_Base):
    """Softmax regression lineare. Parametri: W (d, C) e b (C)."""
    def __init__(self, d, n_classes, wd=0.0):
        self._build_meta((d, n_classes))
        self.wd = wd
    def init(self, rng):
        th = np.zeros(self.N)
        for l in range(self.L):
            sW, (ni, no) = self.blocks[2*l][1], self.blocks[2*l][2]
            th[sW] = rng.normal(0, np.sqrt(1.0/ni), ni*no)
        return th
    def _views(self, th):
        return [th[s].reshape(sh) for _, s, sh in self.blocks]
    def loss_grad(self, th, X, y, out=None):
        (W, b) = self._views(th)
        z = X @ W + b
        zs = z - z.max(1, keepdims=True)
        e = np.exp(zs); s = e.sum(1, keepdims=True)
        n, ar = len(y), np.arange(len(y))
        loss = float(np.mean(np.log(s[:,0]) - zs[ar, y]))
        d = e / s; d[ar, y] -= 1.0
        g = np.empty(self.N) if out is None else out
        g[self.blocks[0][1]] = (X.T @ d / n).ravel()
        g[self.blocks[1][1]] = d.mean(0)
        return loss, g
    def accuracy(self, th, X, y):
        (W, b) = self._views(th)
        return float((np.argmax(X @ W + b, 1) == y).mean())
    def grad_var(self, th, X, y):
        (W, b) = self._views(th)
        z = X @ W + b; zs = z - z.max(1, keepdims=True)
        e = np.exp(zs); s = e.sum(1, keepdims=True)
        d = e / s; d[np.arange(len(y)), y] -= 1.0
        n = len(y)
        gm, gv = np.empty(self.N), np.empty(self.N)
        sW, sb = self.blocks[0][1], self.blocks[1][1]
        mW = X.T @ d / n; sW2 = (X**2).T @ (d**2) / n
        gm[sW], gv[sW] = mW.ravel(), ((sW2 - mW**2) * n/(n-1)).ravel()
        mb = d.mean(0); gm[sb], gv[sb] = mb, ((d**2).mean(0) - mb**2) * n/(n-1)
        return gm, gv
    def proj_grads(self, th, X, y, U):
        (W, b) = self._views(th)
        z = X @ W + b; zs = z - z.max(1, keepdims=True)
        e = np.exp(zs); s = e.sum(1, keepdims=True)
        d = e / s; d[np.arange(len(y)), y] -= 1.0
        n, k = len(y), U.shape[1]
        out = np.zeros((n, k))
        sW, (ni, no) = self.blocks[0][1], self.blocks[0][2]
        T = (X @ U[sW].reshape(ni, no*k)).reshape(n, no, k)
        out += np.einsum("njk,nj->nk", T, d)
        out += d @ U[self.blocks[1][1]]
        return out

class MLP(_Base):
    """MLP con ReLU (o tanh) e softmax. Stessa interfaccia di LinearModel."""
    def __init__(self, sizes, wd=0.0, activation="relu"):
        self._build_meta(sizes); self.wd, self.act = wd, activation
    def init(self, rng):
        th = np.zeros(self.N)
        for l in range(self.L):
            sW, (ni, no) = self.blocks[2*l][1], self.blocks[2*l][2]
            th[sW] = rng.normal(0, np.sqrt(2.0/ni), ni*no)
        return th
    def _views(self, th):
        return [th[s].reshape(sh) for _, s, sh in self.blocks]
    def _forward(self, th, X):
        P = self._views(th); Ws, bs = P[0::2], P[1::2]
        A, Z = [X], []
        for l in range(self.L):
            z = A[-1] @ Ws[l] + bs[l]; Z.append(z)
            A.append(np.maximum(z, 0) if (l < self.L-1 and self.act == "relu") else
                     (np.tanh(z) if (l < self.L-1 and self.act == "tanh") else z))
        return A, Z
    def loss_grad(self, th, X, y, out=None):
        A, Z = self._forward(th, X)
        z = Z[-1]; zs = z - z.max(1, keepdims=True)
        e = np.exp(zs); s = e.sum(1, keepdims=True)
        n, ar = len(y), np.arange(len(y))
        loss = float(np.mean(np.log(s[:,0]) - zs[ar, y]))
        d = e / s; d[ar, y] -= 1.0
        D = [None] * self.L; D[-1] = d
        P = self._views(th); Ws = P[0::2]
        for l in range(self.L-2, -1, -1):
            D[l] = (D[l+1] @ Ws[l+1].T) * (Z[l] > 0 if self.act == "relu" else 1 - np.tanh(Z[l])**2)
        g = np.empty(self.N) if out is None else out
        for l in range(self.L):
            g[self.blocks[2*l][1]] = (A[l].T @ D[l] / n).ravel()
            g[self.blocks[2*l+1][1]] = D[l].mean(0)
        return loss, g
    def accuracy(self, th, X, y, bsz=5000):
        ok = 0
        for i in range(0, len(y), bsz):
            a = X[i:i+bsz]
            A, _ = self._forward(th, a)
            ok += int((A[-1].argmax(1) == y[i:i+bsz]).sum())
        return ok / len(y)
    def grad_var(self, th, X, y):
        A, Z = self._forward(th, X)
        z = Z[-1]; zs = z - z.max(1, keepdims=True)
        e = np.exp(zs); s = e.sum(1, keepdims=True)
        d = e / s; d[np.arange(len(y)), y] -= 1.0
        D = [None]*self.L; D[-1] = d
        P = self._views(th); Ws = P[0::2]
        for l in range(self.L-2, -1, -1):
            D[l] = (D[l+1] @ Ws[l+1].T) * (Z[l] > 0 if self.act == "relu" else 1 - np.tanh(Z[l])**2)
        n = len(y)
        gm, gv = np.empty(self.N), np.empty(self.N)
        for l in range(self.L):
            sW, sb = self.blocks[2*l][1], self.blocks[2*l+1][1]
            mW = A[l].T @ D[l] / n
            sW2 = (A[l]**2).T @ (D[l]**2) / n
            gm[sW], gv[sW] = mW.ravel(), ((sW2 - mW**2) * n/(n-1)).ravel()
            mb = D[l].mean(0); gm[sb], gv[sb] = mb, ((D[l]**2).mean(0) - mb**2) * n/(n-1)
        return gm, gv
    def proj_grads(self, th, X, y, U):
        A, Z = self._forward(th, X)
        z = Z[-1]; zs = z - z.max(1, keepdims=True)
        e = np.exp(zs); s = e.sum(1, keepdims=True)
        d = e / s; d[np.arange(len(y)), y] -= 1.0
        D = [None]*self.L; D[-1] = d
        P = self._views(th); Ws = P[0::2]
        for l in range(self.L-2, -1, -1):
            D[l] = (D[l+1] @ Ws[l+1].T) * (Z[l] > 0 if self.act == "relu" else 1 - np.tanh(Z[l])**2)
        n, k = len(y), U.shape[1]
        out = np.zeros((n, k))
        for l in range(self.L):
            sW, (ni, no) = self.blocks[2*l][1], self.blocks[2*l][2]
            T = (A[l] @ U[sW].reshape(ni, no*k)).reshape(n, no, k)
            out += np.einsum("njk,nj->nk", T, D[l])
            out += D[l] @ U[self.blocks[2*l+1][1]]
        return out

class LinearReg(_Base):
    """Regressione lineare MSE, y = w.x + rumore, senza intercetta (il blocco b e' tenuto a 0: gradiente nullo, init 0).
    Loss = 1/2 mean (w.x - y)^2: e' esattamente il modello della derivazione (Hessiana C, rumore ~ sigma^2 C)."""
    def __init__(self, d, wd=0.0):
        self._build_meta((d, 1)); self.wd = wd
    def init(self, rng):
        th = np.zeros(self.N)
        d = self.sizes[0]
        th[self.blocks[0][1]] = rng.normal(0, np.sqrt(1.0 / d), d)
        return th
    def _w(self, th):
        return th[self.blocks[0][1]]
    def loss_grad(self, th, X, y, out=None):
        r = X @ self._w(th) - y
        g = np.zeros(self.N) if out is None else out
        g[:] = 0.0
        g[self.blocks[0][1]] = X.T @ r / len(y)
        return float(0.5 * np.mean(r ** 2)), g
    def accuracy(self, th, X, y):                      # R^2 (interfaccia compatibile)
        r = X @ self._w(th) - y
        return float(1.0 - np.mean(r ** 2) / max(np.var(y), 1e-30))
    def proj_grads(self, th, X, y, U):
        r = X @ self._w(th) - y
        return (X @ U[self.blocks[0][1]]) * r[:, None]

def make_linear(cfg, data):
    """Modello lineare coerente col task dei dati: MSE per i dati di regressione, softmax altrimenti."""
    if data.get("task") == "reg":
        return LinearReg(cfg.d)
    return LinearModel(cfg.d, cfg.n_classes)

class Curvature:
    """Hessiana della loss (senza weight decay) su un sottoinsieme fisso, via differenze finite."""
    def __init__(self, model, X, y, h=1e-4):
        self.m, self.X, self.y, self.h = model, X, y, h
    def hvp(self, th, v):
        nv = np.linalg.norm(v)
        if nv == 0: return np.zeros_like(v)
        e = self.h / nv
        _, gp = self.m.loss_grad(th + e*v, self.X, self.y)
        _, gm = self.m.loss_grad(th - e*v, self.X, self.y)
        return (gp - gm) / (2*e)
    def top_eigs(self, th, k, rng, tol=1e-3):
        from scipy.sparse.linalg import LinearOperator, eigsh
        N = self.m.N
        if N <= 300:                                   # N piccolo: Hessiana densa, esatta e deterministica
            H = np.column_stack([self.hvp(th, np.eye(N)[:, j]) for j in range(N)])
            w, V = np.linalg.eigh(0.5 * (H + H.T))
            o = np.argsort(w)[::-1][:k]
            return w[o], V[:, o]
        op = LinearOperator((N, N), matvec=lambda v: self.hvp(th, np.asarray(v).ravel()), dtype=np.float64)
        w, V = eigsh(op, k=k, which="LA", tol=tol, v0=rng.normal(size=N), ncv=max(2*k+1, 40))
        o = np.argsort(w)[::-1]
        return w[o], V[:, o]
    def project(self, th, U):
        HU = np.column_stack([self.hvp(th, U[:, j]) for j in range(U.shape[1])])
        Hk = U.T @ HU
        return 0.5 * (Hk + Hk.T)