"""BCGAN core implementation (Bilevel Manifold Fitting, Algorithm 1/2 of the paper).

This module provides a faithful, self-contained implementation of the
probabilistic bilevel CycleGAN for robust manifold fitting:

* Lower level (theta1 = {FM, D_X, D_Y} minimized, theta2 = {G_X, G_Y} maximized):
  the masked MFCGAN objective (Eq.(6) of the paper), i.e. a minimax problem with
  cycle-consistency regularization and mask operations on the ambient inputs
  (x, x_tilde, x_hat) or on the latent outputs (y, y_tilde, y_hat).

* Upper level (s in R^p, Bernoulli relaxation of the binary mask m):
  single time-scale stochastic gradient descent ascent (SSGDA, Algorithm 1) with
  projected policy-gradient estimation (PGE, Eq.(9)-(10)) borrowed from the
  PBCS engineering:
    - REINFORCE score-function gradient  (m - s) / (s * (1 - s));
    - variance reduction with K>=2 mask samples (control variate);
    - exact projection onto the capped simplex C_s = {0<=s<=1, ||s||_1<=C}
      (Algorithm 2) via bisection;
    - decaying stepsizes eta_t, gamma1_t, gamma2_t (inverse to iterations).

Differences w.r.t. PBCS (as required by the BCGAN formulation):
    - the lower problem is a *minimax* problem (SGDA instead of plain descent);
    - the upper-level feedback L_Dmeta(theta*(m)) is the masked cycle objective
      of Eq.(6) evaluated on the meta set through a per-mask adapted copy of
      theta (deep copy + a few Adam rounds = the PBCS "train to convergence"
      inner loop, extended to the minimax lower level), not a supervised loss;
    - mask samples are drawn with the EXACT budget ||m||_0 = C (the discrete
      feasible set of PBCS) instead of relying on the relaxed cap ||s||_1 <= C;
      with variance-reduced REINFORCE (weights sum to zero) the score-function
      estimator remains valid under this size conditioning;
    - masks are applied to data dimensions (noisy dimensions), not to samples.
"""

import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import networks  # noqa: E402
from models.MF import MF as MFModule  # noqa: E402


# ---------------------------------------------------------------------------
# Algorithm 2: projection onto C_s = {s : 0 <= s_i <= 1, ||s||_1 <= C}
# ---------------------------------------------------------------------------
def project_capped_simplex(a, C, tol=1e-7, max_iter=200):
    """P_Cs(a) (Algorithm 2 of the paper).

    Solves 1^T [min(1, max(0, a - b*1))] - C = 0 by bisection, then returns
    min(1, max(0, a - max(0, b)*1)).
    """
    a = a.detach().clone()
    if a.numel() == 0:
        return a
    a_clamped = a.clamp(0.0, 1.0)
    if float(a_clamped.sum()) <= C:
        return a_clamped
    lo, hi = 0.0, float(a.max().item())
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = float(torch.clamp(a - mid, 0.0, 1.0).sum().item())
        if s > C:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    b = max(0.0, 0.5 * (lo + hi))
    return torch.clamp(a - b, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Probabilistic mask (Bernoulli relaxation) and its score function
# ---------------------------------------------------------------------------
def sample_mask_budget(s, C, max_tries=200):
    """Sample m ~ Bern(s) CONDITIONED on ||m||_0 == C (exact budget).

    The paper's constraint is on the mask support size (||m||_0 <= C, the
    discrete feasible set of PBCS).  Conditioning the Bernoulli samples on
    the exact budget C -- rather than relying on the relaxed cap ||s||_1 <= C
    -- (i) rules out degenerate masks such as the all-zero mask, for which
    the masked cycle loss of Eq.(6) vanishes, and (ii) fixes the "carving"
    shortcut observed empirically: with unconstrained sampling the online
    theta re-uses the latent code to carry the noisy dimensions, so dropping
    them *increases* the unmasked reconstruction loss and the feedback
    signal is reversed.  Falls back to the deterministic top-C mask after
    `max_tries` rejected draws.
    """
    C = max(1, min(int(C), int(s.numel())))
    for _ in range(max_tries):
        m = torch.bernoulli(s)
        if int(m.sum().item()) == C:
            return m
    m = torch.zeros_like(s)
    m[torch.topk(s, C).indices] = 1.0
    return m


# ---------------------------------------------------------------------------
# Synthetic data generation (Section 5.2 / Appendix E.1 of the paper)
# ---------------------------------------------------------------------------
NOISY_DIM_STD = 10.0  # noisy dimension follows N(0, 100) (std = 10)


def gen_circle(n, noise_std=0.01, n_noisy_dims=0, seed=0):
    """Eq.(12): X = (cos 2*pi*a, sin 2*pi*a) + 0.01*b, a~U(0,1), b~N(0, I2)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 1, generator=g)
    b = noise_std * torch.randn(n, 2, generator=g)
    x = torch.cat([torch.cos(2 * math.pi * a), torch.sin(2 * math.pi * a)], dim=1) + b
    y = torch.rand(n, 1, generator=g)
    if n_noisy_dims > 0:
        x = torch.cat([x, NOISY_DIM_STD * torch.randn(n, n_noisy_dims, generator=g)], dim=1)
    return x, y


def gen_involute(n, noise_std=0.01, n_noisy_dims=0, seed=0):
    """Eq.(13): X = (a cos 6*pi*a, a sin 6*pi*a) + 0.01*b."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 1, generator=g)
    b = noise_std * torch.randn(n, 2, generator=g)
    x = torch.cat([a * torch.cos(6 * math.pi * a), a * torch.sin(6 * math.pi * a)], dim=1) + b
    y = torch.rand(n, 1, generator=g)
    if n_noisy_dims > 0:
        x = torch.cat([x, NOISY_DIM_STD * torch.randn(n, n_noisy_dims, generator=g)], dim=1)
    return x, y


def gen_torus(n, noise_std=0.01, n_noisy_dims=0, corrupt_output=False, seed=0):
    """Eq.(14): standard torus with R=1, r=0.5.  n_noisy_dims: additive ambient
    noise dims; corrupt_output: append noisy dims N(0,100) to the latent y."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 1, generator=g)
    b = torch.rand(n, 1, generator=g)
    c = noise_std * torch.randn(n, 3, generator=g)
    x1 = (1 + torch.cos(2 * math.pi * a) / 2) * torch.cos(2 * math.pi * b)
    x2 = (1 + torch.cos(2 * math.pi * a) / 2) * torch.sin(2 * math.pi * b)
    x3 = torch.sin(2 * math.pi * a) / 2
    x = torch.cat([x1, x2, x3], dim=1) + c
    y = torch.rand(n, 2, generator=g)
    if corrupt_output and n_noisy_dims > 0:
        y = torch.cat([y, NOISY_DIM_STD * torch.randn(n, n_noisy_dims, generator=g)], dim=1)
    elif n_noisy_dims > 0:
        x = torch.cat([x, NOISY_DIM_STD * torch.randn(n, n_noisy_dims, generator=g)], dim=1)
    return x, y


def make_splits(x, y, m_meta, m_val, m_test, seed=0):
    """Split data into meta / train / val / test sets (i.i.d., Section 4)."""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    i_meta = perm[:m_meta]
    i_val = perm[m_meta:m_meta + m_val]
    i_test = perm[m_meta + m_val:m_meta + m_val + m_test]
    i_train = perm[m_meta + m_val + m_test:]
    return (x[i_meta], y[i_meta]), (x[i_train], y[i_train]), \
           (x[i_val], y[i_val]), (x[i_test], y[i_test])


# ---------------------------------------------------------------------------
# Distances to the latent manifolds (evaluation metrics of Tables 2-3)
# ---------------------------------------------------------------------------
def dist_circle(p):
    return (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1.0).abs()


def dist_torus(p):
    rho = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    return (torch.sqrt((rho - 1.0) ** 2 + p[:, 2] ** 2) - 0.5).abs()


_INVOLUTE_A = torch.linspace(0.0, 1.0, 2001).unsqueeze(1)  # grid on the curve parameter


def dist_involute(p):
    a = _INVOLUTE_A.to(p.device)
    curve = torch.cat([a * torch.cos(6 * math.pi * a), a * torch.sin(6 * math.pi * a)], dim=1)
    d = torch.cdist(p, curve)  # (n, n_grid)
    return d.min(dim=1).values


def manifold_distance(name, points, true_dims):
    p = points[:, true_dims]
    if name == 'circle':
        return dist_circle(p)
    if name == 'involute':
        return dist_involute(p)
    if name == 'torus':
        return dist_torus(p)
    raise ValueError(name)


def metric_summary(dists, sigma=0.01):
    d = dists.detach().cpu().numpy()
    return {
        'mean': float(d.mean()),
        'std': float(d.std()),
        'q95': float(np.quantile(d, 0.95)),
        'pct_over_3sigma': float((d > 3 * sigma).mean() * 100.0),
    }


# ---------------------------------------------------------------------------
# Lower level: masked MFCGAN (Eq.(6)) -- theta1={FM,D_X,D_Y} min, theta2={G_X,G_Y} max
# ---------------------------------------------------------------------------
class LowerMFCGAN(nn.Module):
    """Masked MFCGAN lower-level model.

    Naming follows the paper: G_X: ambient -> latent, G_Y: latent -> ambient,
    D_X: ambient discriminator applied after the manifold fitting module FM,
    D_Y: latent discriminator.

    mask_on='ambient' implements Eq.(6) (mask on x, x_tilde, x_hat);
    mask_on='latent'  implements the symmetric version (mask on y, y_tilde,
    y_hat) used for corrupted latent/output dimensions (e.g. torus).
    """

    def __init__(self, d_ambient, d_latent, X_ref, sigma=0.01,
                 r0=0.05, r1=0.01, r2=0.05, lambda_cycle=10.0,
                 netG='ffnet_9layers', netD='ffnet_9layers', hidden_G=9,
                 gan_mode='lsgan', mask_on='ambient', device=None,
                 meta_d_weight=0.1):
        super().__init__()
        self.d_ambient, self.d_latent = d_ambient, d_latent
        self.sigma = sigma
        self.lambda_cycle = lambda_cycle
        self.mask_on = mask_on
        # weight of the adversarial D-terms inside the upper-level feedback.
        # With a finite inner-level budget theta*(m) is only partially adapted,
        # and the D-terms then carry a systematic bias: a theta trained under
        # keep-all masks scores the FM-denoised pipeline (active exactly when
        # noisy dims are dropped) as "fake".  The masked-vs-unmasked cycle
        # loss (Eq.(5) vs Eq.(6)) carries the true mask-selection signal, so
        # the D-terms are down-weighted by default.
        self.meta_d_weight = meta_d_weight
        self.device = device or torch.device('cpu')

        self.netG_X = networks.define_G([d_ambient], [d_latent], netG, hidden_G, 'none',
                                        False, 'normal', 0.02, [])
        self.netG_Y = networks.define_G([d_latent], [d_ambient], netG, hidden_G, 'none',
                                        False, 'normal', 0.02, [])
        self.netD_Y = networks.define_D([d_latent], netD, 'none', 'normal', 0.02, [])
        self.netD_X = networks.define_D([d_ambient], netD, 'none', 'normal', 0.02, [])
        # manifold fitting module FM: reference points = empirical ambient set
        self.netFM = MFModule(X_ref.to(self.device), r0, r1, r2)
        self.criterionGAN = networks.GANLoss(gan_mode)
        self.to(self.device)

    # -- parameter groups (theta1 = {FM, D_X, D_Y}, theta2 = {G_X, G_Y}) -----
    def theta1_params(self):
        return list(self.netD_X.parameters()) + list(self.netD_Y.parameters()) + \
            list(self.netFM.parameters())

    def theta2_params(self):
        return list(self.netG_X.parameters()) + list(self.netG_Y.parameters())

    # -- one forward pass of the masked pipeline ------------------------------
    def forward_losses(self, x, y, m=None, noise=True):
        """Return the dict of Eq.(6) components (means over the batch).

        m: mask tensor broadcastable to the corrupted variable
           (ambient dims if mask_on='ambient', latent dims otherwise).
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if self.mask_on == 'ambient':
            m_x = m
            m_y = None
        else:
            m_x = None
            m_y = m

        xin = m_x * x if m_x is not None else x                      # m ⊙ x
        fake_y = self.netG_X(xin)                                    # G_X(m ⊙ x)
        yin = m_y * y if m_y is not None else y
        y_for_D_real = (m_y * y) if m_y is not None else y
        fake_y_for_D = (m_y * fake_y) if m_y is not None else fake_y

        # adversarial ambient path: x_tilde = G_Y(y) (+ noise, as in MFCGAN)
        x_tilde = self.netG_Y(yin)
        if noise:
            x_tilde = x_tilde + self.sigma * torch.randn_like(x_tilde)

        # FM projections (Eq.(6): FM applied to m⊙x̃ and m⊙x; the FM reference
        # set is masked consistently via the mask argument)
        mfm = m_x if m_x is not None else None
        fm_fake = self.netFM(m_x * x_tilde if m_x is not None else x_tilde, mask=mfm)
        fm_real = self.netFM(xin, mask=mfm)

        # discriminator outputs
        d_fake_y = self.netD_Y(fake_y_for_D)                         # D_Y on G_X(m⊙x)
        d_real_y = self.netD_Y(y_for_D_real)                         # D_Y on y
        d_fake_x = self.netD_X(fm_fake)                              # D_X ∘ FM (m⊙x̃)
        d_real_x = self.netD_X(fm_real)                              # D_X ∘ FM (m⊙x)

        # cycle-consistency (mask on the corrupted side only, per Eq.(6)).
        # The SAME masked cycle term is reused as the upper-level feedback:
        # with exact-budget mask samples (||m||_0 = C) and per-mask inner
        # adaptation theta*(m) it separates informative from noisy
        # dimensions (see LowerMFCGAN.meta_value).
        bx = self.netG_Y(fake_y)                                     # G_Y ∘ G_X(m⊙x)
        by = self.netG_X(self.netG_Y(yin))                           # G_X ∘ G_Y(...)
        if m_x is not None:
            cycle_A = (m_x * (x - bx)).abs().mean()
            cycle_B = (y - by).abs().mean()
        else:
            cycle_A = (x - bx).abs().mean()
            cycle_B = (m_y * (y - by)).abs().mean()

        return {
            'fake_y': fake_y, 'x_tilde': x_tilde, 'rec_x': bx, 'rec_y': by,
            'd_fake_y': d_fake_y, 'd_real_y': d_real_y,
            'd_fake_x': d_fake_x, 'd_real_x': d_real_x,
            'cycle_A': cycle_A, 'cycle_B': cycle_B,
        }

    # -- theta1 (descent) / theta2 (ascent) losses ----------------------------
    def theta1_loss(self, out):
        """min over {FM, D_X, D_Y}: LSGAN terms of Eq.(6)."""
        return (out['d_fake_y'] ** 2).mean() + ((out['d_real_y'] - 1) ** 2).mean() + \
               (out['d_fake_x'] ** 2).mean() + ((out['d_real_x'] - 1) ** 2).mean()

    def theta2_loss(self, out):
        """The theta2 ascent step of Algorithm 1 implemented as the equivalent
        (and empirically stable) standard generator descent: G wants
        D(fake) -> 1, and minimizes the masked cycle-consistency loss."""
        g_adv = self.criterionGAN(out['d_fake_y'], True) + \
            self.criterionGAN(out['d_fake_x'], True)
        return g_adv + self.lambda_cycle * (out['cycle_A'] + out['cycle_B'])

    def meta_value(self, out):
        """L_Dmeta(theta*(m)) — the upper-level feedback.

        The mask is applied to the pipeline inputs (the deployed denoiser
        b_x = G_Y ∘ G_X(m⊙x)) and the masked cycle-consistency losses of
        Eq.(6) are evaluated on the meta set.  Because the mask samples are
        drawn with the exact budget ||m||_0 = C (never all-zero), the masked
        cycle loss is well defined; combined with the per-mask inner
        adaptation theta*(m) (see BCGANSSGDA._adapt_copy) it separates
        informative from noisy dimensions: masking an informative dim makes
        the masked reconstruction of the kept dims harder, while masking an
        unpredictable noisy dim does not hurt.
        The adversarial D-terms enter with weight meta_d_weight (see
        __init__ for why a small weight is used in practice).
        """
        d_terms = (out['d_fake_y'] ** 2).mean() + ((out['d_real_y'] - 1) ** 2).mean() + \
            (out['d_fake_x'] ** 2).mean() + ((out['d_real_x'] - 1) ** 2).mean()
        return self.meta_d_weight * d_terms + \
            self.lambda_cycle * (out['cycle_A'] + out['cycle_B'])

    @torch.no_grad()
    def eval_forward(self, x, y, m=None):
        was_training = self.training
        self.eval()
        out = self.forward_losses(x, y, m=m, noise=True)
        if was_training:
            self.train()
        return out


# ---------------------------------------------------------------------------
# Upper level + SSGDA bilevel trainer (Algorithm 1)
# ---------------------------------------------------------------------------
class BCGANSSGDA:
    def __init__(self, lower, X_meta, Y_meta, X_train, Y_train,
                 mask_on='ambient', C=None, s_init=0.9,
                 eta=0.05, gamma1=1e-3, gamma2=1e-3,
                 n_inner=5, batch_size=512, K=1, vr=True,
                 adapt_steps=200, adapt_lr=1e-3,
                 clip_grad=3.0, lr_schedule='inverse', lr_decay_horizon=50,
                 C_target=None, ts=0.16, te=0.6, T=200, seed=0,
                 lower_opt='sgd', betas=(0.5, 0.999)):
        self.lower = lower
        self.device = lower.device
        self.mask_on = mask_on
        self.X_meta, self.Y_meta = X_meta.to(self.device), Y_meta.to(self.device)
        self.X_train, self.Y_train = X_train.to(self.device), Y_train.to(self.device)
        p = self.X_meta.shape[1] if mask_on == 'ambient' else self.Y_meta.shape[1]
        self.p = p
        self.C = float(C if C is not None else p)
        # PBCS-style iterative pruning schedule: the cap C_t eases from p to
        # C_target with a cubic ramp between fractions ts and te of T.  This
        # prevents the degenerate all-zero mask (the masked cycle loss vanishes
        # when everything is masked out) by keeping an informative budget.
        self.C_target = float(C_target) if C_target is not None else self.C
        self.ts, self.te = ts, te
        self.T = T
        self.s = torch.full((p,), float(min(s_init, self.C / p)), device=self.device)
        self.s0_init = self.s.clone()
        self.eta0, self.gamma10, self.gamma20 = eta, gamma1, gamma2
        self.n_inner = n_inner
        self.batch_size = batch_size
        self.K = max(1, K)
        self.vr = vr and self.K > 1
        # per-mask inner adaptation (the PBCS "train to convergence" inner
        # loop): each candidate mask is evaluated through a deep copy of the
        # lower level adapted for `adapt_steps` Adam rounds under that mask.
        self.adapt_steps = adapt_steps
        self.adapt_lr = adapt_lr
        self.clip_grad = clip_grad
        self.lr_schedule = lr_schedule
        self.lr_decay_horizon = lr_decay_horizon
        # the lower level uses plain SGD steps in Algorithm 1; Adam with the
        # GAN-standard betas (borrowed from PBCS/MFCGAN engineering) is
        # available because the 9-layer FFN minimax problem barely moves under
        # plain SGD at feasible step sizes.
        if lower_opt == 'adam':
            self.opt_theta1 = torch.optim.Adam(lower.theta1_params(), lr=gamma1, betas=betas)
            self.opt_theta2 = torch.optim.Adam(lower.theta2_params(), lr=gamma2, betas=betas)
        else:
            self.opt_theta1 = torch.optim.SGD(lower.theta1_params(), lr=gamma1)
            self.opt_theta2 = torch.optim.SGD(lower.theta2_params(), lr=gamma2)
        self.history = []
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

    # -- schedules: stepsizes inversely proportional to the iteration count --
    def _lr_factor(self, t):
        if self.lr_schedule == 'inverse':
            return 1.0 / (1.0 + t / self.lr_decay_horizon)
        if self.lr_schedule == 'linear':
            return max(0.0, 1.0 - t / max(1, self.lr_decay_horizon))
        return 1.0

    def _assign_lr(self, t):
        f = self._lr_factor(t)
        for g in self.opt_theta1.param_groups:
            g['lr'] = self.gamma10 * f
        for g in self.opt_theta2.param_groups:
            g['lr'] = self.gamma20 * f
        return self.eta0 * f

    def _sample_batch(self, X, Y, bs):
        idx = self.rng.randint(0, X.shape[0], size=min(bs, X.shape[0]))
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        return X[idx], Y[idx]

    def _mask(self, m):
        if m is None:
            return None
        return m.to(self.device)

    # -- one lower-level SGDA round (Step 1 of Algorithm 1) -------------------
    def lower_step(self, x, y, m):
        out = self.lower.forward_losses(x, y, m=m)
        # theta1 descent (min over FM, D_X, D_Y)
        self.opt_theta1.zero_grad(set_to_none=True)
        loss1 = self.lower.theta1_loss(out)
        loss1.backward()
        self.opt_theta1.step()
        # theta2 ascent (max over G_X, G_Y): implemented as descent of theta2_loss
        self.opt_theta2.zero_grad(set_to_none=True)
        loss2 = self.lower.theta2_loss(self.lower.forward_losses(x, y, m=m))
        loss2.backward()
        self.opt_theta2.step()
        return float(loss1.item()), float(loss2.item())

    def _cap(self, t):
        """Iterative cap schedule C_t (cubic easing, borrowed from PBCS)."""
        if self.C_target >= self.p:
            return self.C
        if t < self.ts * self.T:
            return float(self.p)
        if t < self.te * self.T:
            r = (t - self.ts * self.T) / max(1e-9, (self.te - self.ts) * self.T)
            return self.C_target + (self.p - self.C_target) * (1 - r) ** 3
        return self.C_target

    def _budget(self):
        """Exact mask budget ||m||_0 = C (integer, in [1, p])."""
        return max(1, int(round(min(self.C_target, float(self.p)))))

    def _adapt_copy(self, m, steps=None, lr=None):
        """Deep-copy the lower model and adapt theta*(m) for `steps` Adam
        rounds on the train set — the engineering equivalent of the PBCS
        `train_to_converge` inner loop, extended to the minimax lower level
        (theta1 descent and theta2 descent alternate, as in Algorithm 1).
        The per-mask adaptation removes the systematic bias of evaluating
        all masks through a single online theta (the carving shortcut)."""
        steps = self.adapt_steps if steps is None else steps
        lr = self.adapt_lr if lr is None else lr
        low = copy.deepcopy(self.lower)
        opt1 = torch.optim.Adam(low.theta1_params(), lr=lr, betas=(0.5, 0.999))
        opt2 = torch.optim.Adam(low.theta2_params(), lr=lr, betas=(0.5, 0.999))
        for _ in range(steps):
            x, y = self._sample_batch(self.X_train, self.Y_train, self.batch_size)
            out = low.forward_losses(x, y, m=m)
            opt1.zero_grad(set_to_none=True)
            low.theta1_loss(out).backward()
            opt1.step()
            out = low.forward_losses(x, y, m=m)
            opt2.zero_grad(set_to_none=True)
            low.theta2_loss(out).backward()
            opt2.step()
        return low

    # -- upper-level feedback + projected policy gradient (Step 2, Eq.(9)-(10)) --
    def upper_step(self, t):
        eta_t = self._assign_lr(t)
        cap_t = self._cap(t)
        C = self._budget()
        # each of the K masks is evaluated through its own adapted copy on the
        # FULL meta set: the K feedback values then differ only through the
        # mask, and with the variance-reduction weights w summing to zero the
        # size-conditioning correction of the score function cancels exactly.
        fn_list, grad_list = [], []
        for _ in range(self.K):
            m = sample_mask_budget(self.s, C)
            score = (m - self.s) / (self.s * (1.0 - self.s) + 1e-8)
            m = self._mask(m)
            low = self._adapt_copy(m)
            out = low.forward_losses(self.X_meta, self.Y_meta, m=m)
            fn_list.append(float(low.meta_value(out).item()))
            grad_list.append(score)
        fn_arr = np.asarray(fn_list, dtype=np.float64)
        grad_stack = torch.stack(grad_list, dim=0)  # (K, p)
        if self.vr:
            w = (fn_arr - fn_arr.mean()) / (self.K - 1)
        else:
            w = fn_arr / self.K
        g = (torch.as_tensor(w, dtype=torch.float32, device=self.device).unsqueeze(1)
             * grad_stack).sum(dim=0)
        if self.clip_grad and self.clip_grad > 0:
            g = g / max(1.0, g.norm().item() / self.clip_grad)
        with torch.no_grad():
            self.s.copy_(project_capped_simplex(self.s - eta_t * g, cap_t))
        return eta_t, fn_arr.mean(), g

    # -- full outer iteration --------------------------------------------------
    def step(self, t, log_every=0):
        eta_t = self._assign_lr(t)
        # Step 1 + Step 3: sample a budget-conditioned mask m_t from s_t,
        # update theta (lower level)
        m = sample_mask_budget(self.s, self._budget())
        m = self._mask(m)
        l1, l2 = 0.0, 0.0
        for _ in range(self.n_inner):
            x, y = self._sample_batch(self.X_train, self.Y_train, self.batch_size)
            a, b = self.lower_step(x, y, m)
            l1 += a / self.n_inner
            l2 += b / self.n_inner
        # Step 2: projected policy-gradient update of s with fresh mask samples
        eta_used, fn_mean, _ = self.upper_step(t)
        rec = {'iter': t, 'loss_theta1': l1, 'loss_theta2': l2,
               'meta_loss': fn_mean, 'eta': eta_used,
               's_sum': float(self.s.sum().item()),
               's_mean': float(self.s.mean().item())}
        self.history.append(rec)
        if log_every and (t % log_every == 0):
            print(f"[outer {t:4d}] theta1 {l1:.4f} theta2 {l2:.4f} "
                  f"meta {fn_mean:.4f} eta {eta_t:.2e} |s|_1 {rec['s_sum']:.2f}")
        return rec

    def train(self, T, log_every=10, eval_fn=None, eval_every=0):
        self.T = T
        for t in range(T):
            rec = self.step(t, log_every=log_every)
            if eval_fn is not None and eval_every and ((t + 1) % eval_every == 0 or t == T - 1):
                rec.update(eval_fn(self))
        return self.history

    def hard_mask(self):
        """Discretise s: keep the top-C coordinates (exact budget)."""
        C = self._budget()
        m = torch.zeros_like(self.s)
        m[torch.topk(self.s, C).indices] = 1.0
        return m

    # -- final fine-tuning with the hard mask, as in the paper ----------------
    def finetune_with_hard_mask(self, n_steps=200, batch_size=512, log=0,
                                lr_min_frac=0.02, decay_from=0.5):
        """Constant-then-cosine-decay fine-tuning with the hard mask: the
        decaying step size is what lets the reconstruction reach the high
        precision reported in the paper (a plain constant-lr run plateaus at
        the noise floor)."""
        m = self._mask(self.hard_mask())
        losses = []
        t_decay = max(1, int(decay_from * n_steps))
        for i in range(n_steps):
            if i < t_decay:
                f = 1.0
            else:
                f = lr_min_frac + (1.0 - lr_min_frac) * 0.5 * \
                    (1.0 + math.cos(math.pi * (i - t_decay) / max(1, n_steps - t_decay)))
            for g in self.opt_theta1.param_groups:
                g['lr'] = self.gamma10 * f
            for g in self.opt_theta2.param_groups:
                g['lr'] = self.gamma20 * f
            x, y = self._sample_batch(self.X_train, self.Y_train, batch_size)
            l1, l2 = self.lower_step(x, y, m)
            losses.append((l1, l2))
            if log and (i % log == 0):
                print(f"[finetune {i:4d}] theta1 {l1:.4f} theta2 {l2:.4f}")
        return losses


# ---------------------------------------------------------------------------
# Deployment-stage retraining with a fixed hard mask
# ---------------------------------------------------------------------------
def finetune_lower(lower, X_train, Y_train, m, n_steps=6000, batch_size=512,
                   lr=1e-3, betas=(0.5, 0.999), lr_min_frac=0.02,
                   decay_from=0.5, log=0):
    """Train a lower level with a FIXED hard mask under Adam and a
    constant-then-cosine-decay schedule (the decaying step size is what lets
    the reconstruction reach the high precision reported in the paper; a
    plain constant-lr run plateaus at the noise floor).

    Used for the deployment stage of run_synthetic.py: the bilevel loop
    selects the mask, and the denoising model is then retrained from a FRESH
    initialisation with that mask.  Continuing from the bilevel theta instead
    empirically gets stuck in the bad generator basins left by the
    alternating-mask bilevel phase (the discriminator dominates and the
    generator collapses), which is the GAN analogue of why PBCS-style
    pipelines retrain the final model on the selected support.
    """
    opt1 = torch.optim.Adam(lower.theta1_params(), lr=lr, betas=betas)
    opt2 = torch.optim.Adam(lower.theta2_params(), lr=lr, betas=betas)
    t_decay = max(1, int(decay_from * n_steps))
    losses = []
    for i in range(n_steps):
        if i < t_decay:
            f = 1.0
        else:
            f = lr_min_frac + (1.0 - lr_min_frac) * 0.5 * \
                (1.0 + math.cos(math.pi * (i - t_decay) / max(1, n_steps - t_decay)))
        for g in opt1.param_groups:
            g['lr'] = lr * f
        for g in opt2.param_groups:
            g['lr'] = lr * f
        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        x = X_train[idx].to(lower.device)
        y = Y_train[idx].to(lower.device)
        out = lower.forward_losses(x, y, m=m)
        opt1.zero_grad(set_to_none=True)
        loss1 = lower.theta1_loss(out)
        loss1.backward()
        opt1.step()
        out = lower.forward_losses(x, y, m=m)
        opt2.zero_grad(set_to_none=True)
        loss2 = lower.theta2_loss(out)
        loss2.backward()
        opt2.step()
        losses.append((float(loss1.item()), float(loss2.item())))
        if log and (i % log == 0):
            print(f"[finetune {i:4d}] theta1 {loss1.item():.4f} "
                  f"theta2 {loss2.item():.4f}")
    return losses


# ---------------------------------------------------------------------------
# Evaluation helpers: denoising (x_hat = G_Y∘G_X(m⊙x)) and sample generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def denoise(lower, x, m, batch=4096):
    """Denoised representation x_hat = G_Y ∘ G_X applied to the masked input.
    The mask is applied on the corrupted side: ambient dims for
    mask_on='ambient', the latent code for mask_on='latent'."""
    lower.eval()
    xs = []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(lower.device)
        if lower.mask_on == 'latent':
            y = lower.netG_X(xb)
            ym = m * y if m is not None else y
            xs.append(lower.netG_Y(ym).cpu())
        else:
            mxb = m * xb if m is not None else xb
            xs.append(lower.netG_Y(lower.netG_X(mxb)).cpu())
    return torch.cat(xs, dim=0)


@torch.no_grad()
def generate(lower, n, d_latent, keep_idx=None, batch=4096, seed=0):
    """Feed y ~ U(0,1)^d (masked-out latent dims zeroed) into G_Y."""
    lower.eval()
    g = torch.Generator().manual_seed(seed)
    ys = torch.rand(n, d_latent, generator=g)
    if keep_idx is not None:
        z = torch.zeros_like(ys)
        z[:, keep_idx] = ys[:, keep_idx]
        ys = z
    out = []
    for i in range(0, n, batch):
        out.append(lower.netG_Y(ys[i:i + batch].to(lower.device)).cpu())
    return torch.cat(out, dim=0)
