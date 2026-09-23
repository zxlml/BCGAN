"""Unit and functional tests for the BCGAN bilevel implementation.

Run:
    cd BCGAN && python -m unittest tests.test_bcgan -v
"""

import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

import numpy as np
import torch

import bcgan_core as bc


class TestProjection(unittest.TestCase):
    """Algorithm 2: projection onto the capped simplex."""

    def test_within_region_unchanged(self):
        a = torch.tensor([0.2, 0.5, 0.9])
        out = bc.project_capped_simplex(a, C=10.0)
        self.assertTrue(torch.allclose(out, a.clamp(0, 1)))

    def test_box_constraints_and_cap(self):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(50, generator=g) * 2
        C = 7.5
        out = bc.project_capped_simplex(a, C=C)
        self.assertTrue((out >= 0).all() and (out <= 1).all())
        self.assertLessEqual(float(out.sum()), C + 1e-4)
        # active cap: sum should equal C when the unconstrained sum exceeds it
        if float(a.clamp(0, 1).sum()) > C:
            self.assertAlmostEqual(float(out.sum()), C, places=3)

    def test_extremes(self):
        a = torch.tensor([5.0, -3.0, 2.0])
        self.assertTrue(torch.allclose(bc.project_capped_simplex(a, C=100.0),
                                       torch.tensor([1.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(bc.project_capped_simplex(a, C=0.0),
                                       torch.tensor([0.0, 0.0, 0.0]), atol=1e-5))

    def test_bisection_matches_direct_definition(self):
        g = torch.Generator().manual_seed(1)
        a = torch.rand(10, generator=g) * 3 - 0.5
        out = bc.project_capped_simplex(a, C=3.0)
        # verify optimality: out minimizes ||z - a||^2 over {0<=z<=1, sum<=C}
        def obj(z):
            return float(((z - a) ** 2).sum())
        best = obj(out)
        for _ in range(300):
            z = a.clamp(0, 1)
            i, j = torch.randint(0, 10, (2,), generator=g)
            eps = float(torch.rand(1, generator=g)) * 0.05
            cand = z.clone()
            cand[i] = max(0.0, cand[i] - eps)
            cand[j] = min(1.0, cand[j] + eps)
            if float(cand.sum()) <= 3.0 + 1e-6:
                best = min(best, obj(cand))
        self.assertLessEqual(obj(out), best + 1e-6)


class TestProbabilisticMask(unittest.TestCase):
    def test_bernoulli_mean_and_score_expectation(self):
        g = torch.Generator().manual_seed(0)
        s = torch.tensor([0.3, 0.5, 0.8])
        m = torch.bernoulli(s.repeat(200000, 1), generator=g)
        self.assertTrue(torch.allclose(m.mean(0), s, atol=5e-3))
        score = (m - s) / (s * (1 - s) + 1e-8)
        self.assertTrue(torch.allclose(score.mean(0), torch.zeros(3), atol=5e-2))


class TestReinforceUnbiasedness(unittest.TestCase):
    def test_score_function_gradient_matches_numeric(self):
        """E[f(m) * d ln p / ds] == d/ds E[f(m)] for f(m) = w^T m."""
        torch.manual_seed(3)
        w = torch.tensor([1.5, -0.7, 2.0, 0.3])
        s = torch.tensor([0.4, 0.6, 0.5, 0.7], requires_grad=False)
        N = 400000
        m = torch.bernoulli(s.repeat(N, 1))
        fn = m @ w
        score = (m - s) / (s * (1 - s) + 1e-8)
        est = (fn.unsqueeze(1) * score).mean(0)
        self.assertTrue(torch.allclose(est, w, atol=0.15))


class TestDataGeneration(unittest.TestCase):
    def test_circle_shape_and_noise(self):
        x, y = bc.gen_circle(20000, noise_std=0.01, n_noisy_dims=1, seed=0)
        self.assertEqual(x.shape, (20000, 3))
        self.assertEqual(y.shape, (20000, 1))
        d = bc.dist_circle(x).mean().item()
        self.assertLess(d, 0.03)  # clean circle, sigma=0.01
        self.assertAlmostEqual(x[:, 2].std().item(), 10.0, delta=0.5)

    def test_involute_and_torus(self):
        xi, _ = bc.gen_involute(5000, seed=0)
        self.assertEqual(xi.shape, (5000, 2))
        xt, yt = bc.gen_torus(5000, seed=0)
        self.assertEqual((xt.shape[1], yt.shape[1]), (3, 2))
        d = bc.manifold_distance('torus', xt, [0, 1, 2]).mean().item()
        self.assertLess(d, 0.03)

    def test_corrupted_output_torus(self):
        _, y = bc.gen_torus(5000, n_noisy_dims=1, corrupt_output=True, seed=0)
        self.assertEqual(y.shape[1], 3)
        self.assertAlmostEqual(y[:, 2].std().item(), 10.0, delta=0.5)

    def test_splits_disjoint(self):
        x, y = bc.gen_circle(1000, seed=0)
        a, b, c, d = bc.make_splits(x, y, 200, 100, 100, seed=0)
        n = a[0].shape[0] + b[0].shape[0] + c[0].shape[0] + d[0].shape[0]
        self.assertEqual(n, 1000)


class TestLowerMFCGAN(unittest.TestCase):
    def _build(self):
        """Use realistic circle data so that the FM reference set has
        neighbors within r0 (the sparse-random case falls back to identity)."""
        torch.manual_seed(0)
        x, y = bc.gen_circle(200, noise_std=0.01, n_noisy_dims=1, seed=0)
        low = bc.LowerMFCGAN(3, 1, x, sigma=0.01, mask_on='ambient')
        xs, ys = bc.gen_circle(64, noise_std=0.01, n_noisy_dims=1, seed=7)
        return low, xs, ys

    def test_forward_shapes_and_finiteness(self):
        low, x, y = self._build()
        out = low.forward_losses(x, y, m=torch.ones(3))
        for k in ['d_fake_y', 'd_real_y', 'd_fake_x', 'd_real_x']:
            self.assertEqual(out[k].shape[0], 64)
        for k, v in out.items():
            self.assertTrue(torch.isfinite(v).all(), k)

    def test_fm_identity_fallback(self):
        torch.manual_seed(0)
        X_ref = torch.randn(20, 3) * 0.3 + 10.0  # far away from queries
        low = bc.LowerMFCGAN(3, 1, X_ref, mask_on='ambient')
        z = torch.randn(8, 3)
        out = low.netFM(z)
        self.assertTrue(torch.allclose(out, z))

    def test_mask_changes_fm_reference(self):
        low, x, y = self._build()
        m_full = torch.ones(3)
        m_noisy = torch.tensor([1.0, 1.0, 0.0])
        # with the noisy dim active the reference cloud is spread over ~10 in
        # that coordinate, so FM has no neighbours within r0 and falls back to
        # the identity; masking it enables the 2D manifold fit.  Compare at
        # the FM level (the freshly initialised D_X has tiny gain and would
        # hide the difference in its output).
        z_full = low.netFM(x, mask=m_full)
        z_noisy = low.netFM(m_noisy * x, mask=m_noisy)
        self.assertFalse(torch.allclose(z_full, z_noisy))
        self.assertTrue(torch.allclose(z_noisy[:, :2], (m_noisy * x)[:, :2], atol=0.5))

    def test_theta1_grads_flow(self):
        low, x, y = self._build()
        # drop the noisy dim so that FM actually finds neighbours within r0
        # (with the noisy dim active the reference cloud is spread over ~10
        # in that coordinate and FM falls back to the identity map)
        out = low.forward_losses(x, y, m=torch.tensor([1.0, 1.0, 0.0]))
        low.theta1_loss(out).backward()
        has_g = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in low.netD_X.parameters())
        has_mf = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in low.netFM.parameters())
        self.assertTrue(has_g and has_mf)

    def test_theta2_descent_reduces_generator_loss(self):
        low, x, y = self._build()
        m = torch.ones(3)
        opt = torch.optim.SGD(low.theta2_params(), lr=1e-3)
        first = None
        for i in range(60):
            out = low.forward_losses(x, y, m=m)
            loss = low.theta2_loss(out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss.item())
        self.assertLess(float(loss.item()), first)


class TestSSGDA(unittest.TestCase):
    def _solve(self, mask_on='ambient', T=40):
        torch.manual_seed(0)
        if mask_on == 'ambient':
            x, y = bc.gen_circle(1200, n_noisy_dims=1, seed=0)
            d0, d = 3, 1
        else:
            x, y = bc.gen_torus(1200, n_noisy_dims=1, corrupt_output=True, seed=0)
            d0, d = 3, 3
        (Xm, Ym), (Xt, Yt), _, _ = bc.make_splits(x, y, 300, 100, 100, seed=0)
        low = bc.LowerMFCGAN(d0, d, Xt, sigma=0.01, mask_on=mask_on)
        solver = bc.BCGANSSGDA(low, Xm, Ym, Xt, Yt, mask_on=mask_on,
                               eta=0.1, gamma1=2e-3, gamma2=2e-3,
                               n_inner=2, batch_size=256,
                               K=2, vr=True, adapt_steps=4, adapt_lr=1e-3,
                               C_target=d0 - 1,
                               ts=0.1, te=0.5, seed=0)
        solver.train(T, log_every=0)
        return solver

    def test_smoke_ambient(self):
        solver = self._solve('ambient', T=30)
        self.assertTrue(torch.isfinite(solver.s).all())
        self.assertGreaterEqual(float(solver.s.sum()), 0.0)
        self.assertEqual(len(solver.history), 30)

    def test_smoke_latent(self):
        solver = self._solve('latent', T=20)
        self.assertTrue(torch.isfinite(solver.s).all())

    def test_meta_feedback_identifies_noisy_dim(self):
        """Functional check of the upper-level feedback (Eq.(5)): after the
        lower level is trained on the true mask, L_Dmeta must be smallest
        when the noisy dim is masked and larger when an informative dim is
        masked or everything is kept."""
        torch.manual_seed(0)
        x, y = bc.gen_circle(2000, n_noisy_dims=1, seed=0)
        (Xm, Ym), (Xt, Yt), _, _ = bc.make_splits(x, y, 400, 1200, 100, seed=0)
        low = bc.LowerMFCGAN(3, 1, Xt, sigma=0.01, mask_on='ambient')
        m_true = torch.tensor([1.0, 1.0, 0.0])
        g = torch.Generator().manual_seed(1)
        opt1 = torch.optim.Adam(low.theta1_params(), lr=1e-3, betas=(0.5, 0.999))
        opt2 = torch.optim.Adam(low.theta2_params(), lr=1e-3, betas=(0.5, 0.999))
        for _ in range(600):
            idx = torch.randint(0, Xt.shape[0], (256,), generator=g)
            out = low.forward_losses(Xt[idx], Yt[idx], m=m_true)
            opt1.zero_grad(); low.theta1_loss(out).backward(); opt1.step()
            out = low.forward_losses(Xt[idx], Yt[idx], m=m_true)
            opt2.zero_grad(); low.theta2_loss(out).backward(); opt2.step()
        with torch.no_grad():
            out_noisy = low.meta_value(low.forward_losses(Xm, Ym, m=m_true)).item()
            out_x0 = low.meta_value(
                low.forward_losses(Xm, Ym, m=torch.tensor([0., 1., 1.]))).item()
            out_keep = low.meta_value(low.forward_losses(Xm, Ym, m=torch.ones(3))).item()
        self.assertLess(out_noisy, out_x0)
        self.assertLess(out_noisy, out_keep)

    def test_projection_keeps_cap(self):
        solver = self._solve('ambient', T=10)
        solver.C_target = 1.5
        for t in range(10):
            solver.upper_step(t)
            # the PBCS easing schedule prescribes the cap at every t
            self.assertLessEqual(float(solver.s.sum()), solver._cap(t) + 1e-4)
            self.assertTrue(((solver.s >= 0) & (solver.s <= 1)).all())
        # after the ramp the cap must be exactly C_target
        solver.upper_step(10)
        self.assertLessEqual(float(solver.s.sum()), 1.5 + 1e-4)

    def test_lr_decay(self):
        solver = self._solve('ambient', T=1)
        f0 = solver._lr_factor(0)
        f100 = solver._lr_factor(100)
        self.assertAlmostEqual(f0, 1.0)
        self.assertLess(f100, f0)


class TestEvaluationHelpers(unittest.TestCase):
    def test_metrics_summary(self):
        d = torch.tensor([0.001] * 90 + [0.05] * 10)
        s = bc.metric_summary(d)
        self.assertAlmostEqual(s['mean'], 0.0059, places=4)
        self.assertAlmostEqual(s['pct_over_3sigma'], 10.0, places=6)

    def test_denoise_generate_shapes(self):
        torch.manual_seed(0)
        x, y = bc.gen_circle(600, n_noisy_dims=1, seed=0)
        (Xm, Ym), (Xt, Yt), _, (Xs, Ys) = bc.make_splits(x, y, 150, 100, 100, seed=0)
        low = bc.LowerMFCGAN(3, 1, Xt, mask_on='ambient')
        m = torch.tensor([1.0, 1.0, 0.0])
        xd = bc.denoise(low, Xs, m)
        xg = bc.generate(low, 50, 1)
        self.assertEqual(xd.shape, Xs.shape)
        self.assertEqual(xg.shape, (50, 3))
        self.assertTrue(torch.isfinite(xd).all() and torch.isfinite(xg).all())


if __name__ == '__main__':
    unittest.main()
