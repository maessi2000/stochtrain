"""config.py — parametri centralizzati. Un solo posto da toccare."""
from dataclasses import dataclass, replace

@dataclass
class Config:
    # dati
    dataset: str = "synthetic"        # synthetic | digits | mnist | cifar10
    data_dir: str = "data"
    d: int = 10                       # dimensione input (sintetico)
    n_train: int = 20000
    n_test: int = 2000
    spectrum: tuple = (2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001)
    sigma: float = 1.0
    seed_data: int = 0
    teacher: str = "linear"           # linear | mlp : insegnante dei dati sintetici di classificazione
    c0_model: str = "mse"             # mse | softmax : modello lineare di C0/C3 (mse = caso esattamente risolvibile)

    # modello lineare (stessa architettura del sintetico)
    n_classes: int = 2

    # modello MLP
    mlp_sizes: tuple = (10, 32, 32, 2)   # solo per sintetico; per digits/MNIST si adatta
    activation: str = "relu"

    # SGD
    lr: float = 0.01
    bs: int = 16
    wd: float = 1e-3
    n_burn: int = 0                   # 0 -> burn_taus / (lr*wd)
    burn_taus: float = 5.0            # burn-in in unita' del tempo di rilassamento lento 1/(lr*wd)
    n_pilot: int = 20000              # passi per stimare il centro theta_bar dopo il burn-in
    n_stat: int = 40000
    chunk: int = 200
    eval_every: int = 20              # in chunk
    snap_every: int = 50              # snapshot di theta nella finestra

    # spettro / sottospazio
    hess_k: int = 10
    hess_n: int = 2000
    grad_n: int = 2000

    # attribuzioni
    attr_method: str = "grad_x_input" # linear | grad_x_input | ig
    attr_points: int = 20
    ig_steps: int = 50

    # griglia scaling (C1)
    scaling_lrs: tuple = (0.005, 0.01, 0.02)
    scaling_bs: tuple = (4, 16, 64)
    scaling_taus: float = 3.0         # rilassamento e finestra dello scaling = scaling_taus * tau_max(lr)
    scaling_cap: int = 200000         # tetto ai passi per configurazione

    # C3
    n_seeds_erg: int = 20
    erg_train_lengths: tuple = (5000, 10000, 20000, 40000)

    # presets
    @property
    def burn(self) -> int:
        n = self.n_burn or (int(self.burn_taus / (self.lr * self.wd)) if self.wd > 0 else 20000)
        return max(self.chunk, (n // self.chunk) * self.chunk)
    @property
    def stat_steps(self) -> int:
        return max(self.chunk, (self.n_stat // self.chunk) * self.chunk)

PRESETS = {
    "smoke": dict(
        n_burn=4000, n_stat=8000, chunk=200, n_pilot=2000,
        scaling_taus=2.0, scaling_cap=8000,
        hess_k=6, hess_n=800, grad_n=800,
        attr_points=5,
        scaling_lrs=(0.001, 0.005, 0.01, 0.02),
        scaling_bs=(4, 8, 16),
        n_seeds_erg=5,
        erg_train_lengths=(2000, 4000),
    ),

    "core": dict(
        # training / finestra stazionaria: burn = 5/(lr*wd) (auto), finestra >~ 5 tau_max
        n_burn=0,
        n_stat=260000,
        chunk=200,
        eval_every=20,
        snap_every=25,

        # spettro / sottospazio
        hess_k=10,
        hess_n=2000,
        grad_n=2000,

        # attribuzioni
        attr_method="grad_x_input",
        attr_points=20,
        ig_steps=50,

        # griglia scaling C1
        scaling_lrs=(0.005, 0.01, 0.02),
        scaling_bs=(4, 16, 64),

        # C3
        n_seeds_erg=20,
        erg_train_lengths=(20000, 40000, 80000, 160000),
    ),
}