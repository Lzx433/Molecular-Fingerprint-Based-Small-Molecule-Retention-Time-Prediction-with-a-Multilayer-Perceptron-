"""Four-panel measured-versus-predicted plot using real held-out predictions.

Uses seed 17, shows every test molecule, and recomputes the displayed metrics
from the same points. No retraining, filtering, or axis clipping is performed.
"""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.runtime').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/logs/matplotlib_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    arrangements = [
        ('random', 'rf', '#7253a3', 'A', 'Random split / Random forest'),
        ('random', 'mlp', '#078480', 'B', 'Random split / MLP'),
        ('scaffold', 'rf', '#bd7b18', 'C', 'Scaffold split / Random forest'),
        ('scaffold', 'mlp', '#bf2852', 'D', 'Scaffold split / MLP'),
    ]
    runs = []
    for protocol, model, color, letter, title in arrangements:
        data = pd.read_csv(root / f'outputs/predictions/{protocol}_{model}_seed17.csv')
        split = pd.read_csv(root / f'splits/{protocol}.csv')
        test_ids = set(split.loc[split.split == 'test', 'row_id'])
        assert not data.row_id.duplicated().any()
        assert set(data.row_id) == test_ids
        y, pred = data.y_true.to_numpy(), data.y_pred.to_numpy()
        assert np.isfinite(y).all() and np.isfinite(pred).all()
        err = pred - y
        metrics = dict(protocol=protocol, model=model, seed=17, n=len(y),
                       MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                       MedAE=float(np.median(np.abs(err))),
                       R2=float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        runs.append((data, metrics, color, letter, title))
    all_values = np.concatenate([d[['y_true', 'y_pred']].to_numpy().ravel() for d, *_ in runs])
    lower = min(0, np.floor(all_values.min() / 150) * 150)
    upper = np.ceil(all_values.max() / 150) * 150
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 12, 'axes.linewidth': .9, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (data, metrics, color, letter, title) in zip(axes.ravel(), runs):
        ax.scatter(data.y_true, data.y_pred, s=11, facecolors='none', edgecolors=color,
                   linewidths=.45, alpha=.52, label=f'Test molecules (n = {len(data):,})')
        ax.plot([lower, upper], [lower, upper], color='#d82c32', linestyle='--',
                linewidth=1.15, label='Ideal prediction: y = x')
        ax.set(xlim=(lower, upper), ylim=(lower, upper),
               xlabel='Experimental retention time (s)', ylabel='Predicted retention time (s)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(0, upper + 1, 300))
        ax.set_yticks(np.arange(0, upper + 1, 300))
        ax.tick_params(direction='in', labelsize=10)
        ax.set_title(f'({letter}) {title}', fontsize=14, pad=12)
        legend = ax.legend(loc='lower right', fontsize=9, frameon=False, handlelength=2, markerscale=1.6)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        # Metrics are outside the plotting area, so even large-error points
        # near the upper left remain unobscured by a text box.
        line = (f"$R^2$ = {metrics['R2']:.4f}    RMSE = {metrics['RMSE']:.2f} s\n"
                f"MAE = {metrics['MAE']:.2f} s    MedAE = {metrics['MedAE']:.2f} s")
        ax.text(.5, -.17, line, transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color='#163da5', linespacing=1.6)
    fig.suptitle('METLIN SMRT: model comparison on independent test sets', fontsize=18, y=.985)
    fig.subplots_adjust(left=.085, right=.98, top=.93, bottom=.17, wspace=.27, hspace=.48)
    fig.text(.5, .024, 'Representative runs: seed 17. All test points are shown, including negative predictions.\n'
             'Each panel reports its own test metrics; the two split protocols contain different test molecules.',
             ha='center', va='bottom', fontsize=10, linespacing=1.5)
    figures = root / 'outputs/figures'
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(figures / f'model_comparison_reference_style.{extension}', dpi=300, facecolor='white')
    pd.DataFrame([record[1] for record in runs]).to_csv(
        root / 'outputs/tables/model_comparison_reference_style_metrics.csv', index=False)
    plt.close(fig)
    print('Created the four-panel test-set figure and its metric table.')


if __name__ == '__main__':
    main()
