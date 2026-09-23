"""Run BCGAN (Algorithm 1, SSGDA + projected policy gradient) on the paper's
synthetic benchmarks (Section 5.2, Eq.(12)-(14)) under clean and corrupted
(noisy dimension N(0,100)) scenarios, and report the Table 2/3 metrics:
mean +/- SD, 95% quantile, and percentage of samples beyond 3*sigma of the
latent manifold (sigma = 0.01).

Usage (CPU, e.g.):
    python run_synthetic.py --dataset circle  --corrupt input
    python run_synthetic.py --dataset involute --corrupt input
    python run_synthetic.py --dataset torus   --corrupt output
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bcgan_core as bc

PAPER_REF = {  # Table 2 / Table 3 "BCGAN (Ours)" mean distances
    ('circle', 'none'): 7.21e-4,
    ('circle', 'input'): 7.49e-4,
    ('involute', 'none'): 5.08e-3,
    ('involute', 'input'): 5.17e-3,
    ('torus', 'none'): 6.37e-3,
    ('torus', 'output'): 6.44e-3,
}


def build_args():
    p = argparse.ArgumentParser(description='BCGAN synthetic experiments')
    p.add_argument('--dataset', type=str, default='circle',
                   choices=['circle', 'involute', 'torus'])
    p.add_argument('--corrupt', type=str, default='input',
                   choices=['none', 'input', 'output'],
                   help="where the noisy dimension N(0,100) is appended")
    p.add_argument('--n_noisy', type=int, default=1)
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--m1', type=int, default=1000, help='meta set size')
    p.add_argument('--m_val', type=int, default=500)
    p.add_argument('--m_test', type=int, default=500)
    p.add_argument('--T', type=int, default=12, help='outer iterations')
    p.add_argument('--n_inner', type=int, default=20, help='lower SGDA rounds per outer iter')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--K', type=int, default=2, help='budget-conditioned mask samples for the PGE')
    p.add_argument('--eta', type=float, default=0.05, help='upper step size (s)')
    p.add_argument('--adapt_steps', type=int, default=200,
                   help='inner Adam rounds adapting a copy of theta per candidate mask')
    p.add_argument('--adapt_lr', type=float, default=1e-3,
                   help='learning rate of the per-mask inner adaptation')
    p.add_argument('--meta_d_weight', type=float, default=0.1,
                   help='weight of the adversarial D-terms in the upper feedback')
    p.add_argument('--gamma1', type=float, default=1e-3, help='lower step size (theta1: FM,D)')
    p.add_argument('--gamma2', type=float, default=1e-3, help='lower step size (theta2: G)')
    p.add_argument('--lower_opt', type=str, default='adam', choices=['sgd', 'adam'],
                   help='lower-level optimizer (Adam with GAN betas in practice)')
    p.add_argument('--lambda_cycle', type=float, default=10.0)
    p.add_argument('--sigma', type=float, default=0.01)
    p.add_argument('--C', type=float, default=0.0, help='cap of ||s||_1; 0 -> p (default)')
    p.add_argument('--C_target', type=float, default=0.0,
                   help='final budget of the iterative pruning; 0 -> auto (p - n_noisy)')
    p.add_argument('--ts', type=float, default=0.3, help='pruning start (fraction of T)')
    p.add_argument('--te', type=float, default=0.7, help='pruning end (fraction of T)')
    p.add_argument('--s_init', type=float, default=0.9)
    p.add_argument('--lr_schedule', type=str, default='inverse',
                   choices=['inverse', 'linear', 'const'])
    p.add_argument('--finetune', type=int, default=6000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save', type=str, default='')
    p.add_argument('--plot', action='store_true')
    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- data (Section 5.2) ---------------------------------
    noise_dims = args.n_noisy if args.corrupt != 'none' else 0
    if args.dataset == 'circle':
        x, y = bc.gen_circle(args.n, args.sigma, noise_dims, seed=args.seed)
        true_amb_dims, true_lat_dims = [0, 1], [0]
        mask_on = 'ambient'
    elif args.dataset == 'involute':
        x, y = bc.gen_involute(args.n, args.sigma, noise_dims, seed=args.seed)
        true_amb_dims, true_lat_dims = [0, 1], [0]
        mask_on = 'ambient'
    else:
        co = args.corrupt == 'output'
        x, y = bc.gen_torus(args.n, args.sigma, noise_dims, corrupt_output=co, seed=args.seed)
        true_amb_dims = [0, 1, 2]
        true_lat_dims = [0, 1] if co else [0, 1]
        mask_on = 'latent' if co else 'ambient'
    d0, d = x.shape[1], y.shape[1]

    (Xm, Ym), (Xt, Yt), (Xv, Yv), (Xs, Ys) = bc.make_splits(
        x, y, args.m1, args.m_val, args.m_test, seed=args.seed)
    print(f'dataset={args.dataset} corrupt={args.corrupt} ambient d0={d0} latent d={d} '
          f'meta={Xm.shape[0]} train={Xt.shape[0]} device={device}')

    # ---------------- lower level: masked MFCGAN --------------------------
    lower = bc.LowerMFCGAN(d0, d, Xt, sigma=args.sigma, lambda_cycle=args.lambda_cycle,
                           mask_on=mask_on, device=device,
                           meta_d_weight=args.meta_d_weight)

    # ---------------- bilevel SSGDA ----------------------------------------
    p_dim = d0 if mask_on == 'ambient' else d
    auto_C = max(1.0, float(p_dim - noise_dims))
    solver = bc.BCGANSSGDA(lower, Xm, Ym, Xt, Yt,
                           mask_on=mask_on,
                           C=args.C if args.C > 0 else None,
                           s_init=args.s_init,
                           eta=args.eta, gamma1=args.gamma1, gamma2=args.gamma2,
                           n_inner=args.n_inner, batch_size=args.batch_size,
                           K=args.K, adapt_steps=args.adapt_steps,
                           adapt_lr=args.adapt_lr,
                           lr_schedule=args.lr_schedule,
                           C_target=(args.C_target if args.C_target > 0 else auto_C),
                           ts=args.ts, te=args.te, T=args.T, seed=args.seed,
                           lower_opt=args.lower_opt)

    val_batch = (Xv[:512], Yv[:512])
    test_batch = (Xs[:512], Ys[:512])

    def eval_fn(sl):
        m_hard = sl.hard_mask().to(sl.device)
        with torch.no_grad():
            lv = sl.lower.meta_value(sl.lower.forward_losses(*val_batch, m=m_hard))
            ls = sl.lower.meta_value(sl.lower.forward_losses(*test_batch, m=m_hard))
        return {'val_loss': float(lv.item()), 'test_loss': float(ls.item()),
                'gen_gap': float(abs(lv.item() - ls.item()))}

    t0 = time.time()
    solver.train(args.T, log_every=10, eval_fn=eval_fn, eval_every=10)
    print(f'outer loop done in {time.time() - t0:.1f}s; hard mask = '
          f'{solver.hard_mask().int().tolist()}')

    if args.finetune > 0:
        solver.finetune_with_hard_mask(args.finetune, args.batch_size)

    # ---------------- evaluation (Tables 2-3 metrics) ----------------------
    m_hard = solver.hard_mask().to(device)
    x_denoised = bc.denoise(lower, Xs, m_hard)  # mode-aware: ambient or latent mask
    keep = torch.nonzero(m_hard).flatten().tolist() if mask_on == 'latent' else None
    x_generated = bc.generate(lower, Xs.shape[0], d, keep_idx=keep, seed=args.seed + 1)

    m_denoise = bc.manifold_distance(args.dataset, x_denoised, true_amb_dims)
    m_gen = bc.manifold_distance(args.dataset, x_generated, true_amb_dims)
    s_denoise, s_gen = bc.metric_summary(m_denoise), bc.metric_summary(m_gen)

    # input baseline
    m_input = bc.manifold_distance(args.dataset, Xs, true_amb_dims)
    s_input = bc.metric_summary(m_input)

    ref = PAPER_REF.get((args.dataset, args.corrupt))
    print('\n===== Metrics: distance to the latent manifold =====')
    print(f"{'source':<12}{'mean±SD':<24}{'95% quantile':<16}{'>3σ':<10}")
    print(f"{'input':<12}{s_input['mean']:.4e} ± {s_input['std']:.3e}   "
          f"{s_input['q95']:.4e}   {s_input['pct_over_3sigma']:.2f}%")
    print(f"{'denoised':<12}{s_denoise['mean']:.4e} ± {s_denoise['std']:.3e}   "
          f"{s_denoise['q95']:.4e}   {s_denoise['pct_over_3sigma']:.2f}%")
    print(f"{'generated':<12}{s_gen['mean']:.4e} ± {s_gen['std']:.3e}   "
          f"{s_gen['q95']:.4e}   {s_gen['pct_over_3sigma']:.2f}%")
    if ref is not None:
        print(f"paper (BCGAN, Table 2/3): mean ≈ {ref:.3e}")

    last = solver.history[-1] if solver.history else {}
    print(f"final meta(train) loss {last.get('meta_loss', float('nan')):.4f}, "
          f"val {solver.history[-1].get('val_loss', float('nan')) if solver.history else float('nan'):.4f}, "
          f"test {solver.history[-1].get('test_loss', float('nan')) if solver.history else float('nan'):.4f}, "
          f"gap {solver.history[-1].get('gen_gap', float('nan')) if solver.history else float('nan'):.4f}")

    # ---------------- save artifacts ---------------------------------------
    out_dir = args.save or os.path.join('results', 'bcgan',
                                        f'{args.dataset}_{args.corrupt}_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'mean', 'std', 'q95', 'pct_over_3sigma'])
        for name, s in [('input', s_input), ('denoised', s_denoise), ('generated', s_gen)]:
            w.writerow([name, s['mean'], s['std'], s['q95'], s['pct_over_3sigma']])
        w.writerow(['paper_ref', ref, '', '', ''])
    with open(os.path.join(out_dir, 'history.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(solver.history[0].keys() if solver.history else ['iter'])
        for rec in solver.history:
            w.writerow(rec.values())
    np.savetxt(os.path.join(out_dir, 'scores.csv'), solver.s.cpu().numpy())
    torch.save({'lower': lower.state_dict(), 'scores': solver.s.cpu(),
                'mask': solver.hard_mask().cpu()}, os.path.join(out_dir, 'model.pt'))
    print(f"artifacts saved to {out_dir}")

    # ---------------- optional scatter plot ---------------------------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            xd = x_denoised.numpy()
            axes[0].scatter(Xs[:, 0], Xs[:, 1], s=2, alpha=0.3)
            axes[0].set_title('noisy input')
            axes[1].scatter(xd[:, 0], xd[:, 1], s=2, c='g', alpha=0.5)
            axes[1].set_title('BCGAN denoised')
            axes[2].scatter(x_generated[:, 0], x_generated[:, 1], s=2, c='r', alpha=0.5)
            axes[2].set_title('BCGAN generated')
            for a in axes:
                a.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, 'scatter.png'), dpi=150)
            print('scatter saved')
        except Exception as e:  # matplotlib missing is fine
            print('plot skipped:', e)


if __name__ == '__main__':
    main()
