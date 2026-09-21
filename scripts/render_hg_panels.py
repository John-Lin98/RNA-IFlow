"""Render R7 Figure 2(d,e) from packaged quality and cost data.

Adapted plotting geometry/colors from render_final.py SHA256
28d33e3ef0d3fa14bf7a2b36bfd23b7b419dc790f36499a329ee1d6eb0b4155b.
Uses portable DejaVu Sans, not a claim of pixel-identical Arial rendering.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'results/figures/source_data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    quality_path = args.source / 'hg/HG_mean_sd.csv'
    cost_path = args.source / 'hg_cost/HG_cost.csv'
    with quality_path.open() as handle:
        quality = list(csv.DictReader(handle))
    with cost_path.open() as handle:
        costs = list(csv.DictReader(handle))
    panels = []
    for axis, grid, letter in [('H', [1, 2, 4, 8, 16], 'd'), ('G', [2, 4, 8, 16], 'e')]:
        means, errors, seconds = [], [], []
        for value in grid:
            rows = [r for r in quality if int(r[axis]) == value and int(r['G' if axis == 'H' else 'H']) == 8]
            c = [r for r in costs if r['axis'] == axis and int(r['value']) == value]
            if len(rows) != 1 or int(rows[0]['n']) != 3 or len(c) != 1:
                raise ValueError('Incomplete or duplicate H/G point')
            mean, error, duration = float(rows[0]['pass_at_8_mean']), float(rows[0]['pass_at_8_sd']), float(c[0]['seconds'])
            if not all(math.isfinite(x) for x in (mean, error, duration)) or not 0 <= mean <= 1 or min(error, duration) < 0:
                raise ValueError('Invalid plot value')
            means.append(mean)
            errors.append(error)
            seconds.append(duration)
        panels.append((axis, grid, letter, means, errors, seconds))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
    args.output.mkdir(parents=True)
    outputs = []
    for axis, grid, letter, means, errors, seconds in panels:
        fig = plt.figure(figsize=(2.70, 1.80))
        ax = fig.add_axes((.215, .275, .555, .515))
        right = ax.twinx()
        right.spines['right'].set_visible(True)
        fig.text(.015, .976, f'({letter})', ha='left', va='top', fontsize=9, fontweight='bold')
        right.bar(range(len(grid)), seconds, width=.48, color='#DFE8EE', edgecolor='#C7D3DD', linewidth=.5)
        ax.set_zorder(right.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(range(len(grid)), means, 'o-', color='#0072B2', ms=4, lw=1.1, mfc='white', zorder=5)
        ax.errorbar(range(len(grid)), means, yerr=errors, fmt='none', ecolor='#0072B2',
                    elinewidth=1, capsize=3.8, capthick=1, zorder=6)
        ax.set_xticks(range(len(grid)), grid)
        ax.set_xlim(-.5, len(grid) - .5)
        ax.set_ylim(.58, .71)
        ax.set_yticks([.58, .62, .66, .70])
        ax.set_xlabel('Policy steps H' if axis == 'H' else 'Group size G')
        ax.set_ylabel('P@8')
        right.set_ylim(0, 1.7 if axis == 'H' else 70)
        right.set_yticks([0, .5, 1, 1.5] if axis == 'H' else [0, 20, 40, 60])
        right.set_ylabel('Inference time (s)' if axis == 'H' else 'Update time (s)', color='#536774', fontsize=8)
        right.tick_params(colors='#536774', labelsize=8)
        fig.legend([Line2D([], [], color='#0072B2', marker='o', lw=1, ms=4),
                    Patch(facecolor='#DFE8EE', edgecolor='#C7D3DD')], ['P@8 ± SD', 'Time (s)'],
                   loc='upper center', bbox_to_anchor=(.53, 1.005), frameon=False, ncol=2,
                   fontsize=7.8, columnspacing=.8, handlelength=1.6)
        for suffix in ('pdf', 'png'):
            path = args.output / f'Figure2_panel_{letter}.{suffix}'
            fig.savefig(path, dpi=300)
            outputs.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plt.close(fig)
    manifest = dict(status='rendered', scope='H/G panels only; not complete Figure 2 or final manuscript approval',
                    font='DejaVu Sans (portable substitution)', matplotlib=matplotlib.__version__,
                    sd='sample SD across three continuation training seeds',
                    inputs=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (quality_path, cost_path)],
                    outputs=outputs)
    (args.output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PASS: H/G panels rendered from packaged data')


if __name__ == '__main__':
    main()
