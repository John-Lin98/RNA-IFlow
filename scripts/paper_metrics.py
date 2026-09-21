"""Paper scoring adapter; keep the historical training scorer unchanged."""
import math
from pathlib import Path
import sys

import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/rna-flow-fair-components'))
from evaluate import evaluate_candidate as historical_evaluate_candidate
from constraints import target_pairs


def evaluate_candidate(sequence, target):
    if RNA.__version__ != '2.7.2':
        raise RuntimeError('The frozen paper protocol requires ViennaRNA 2.7.2')
    if not target or set(target) - set('.()') or len(sequence) != len(target) or set(sequence) - set('ACGU'):
        raise ValueError('Expected equal-length RNA and dot-bracket target')
    target_pairs(target)
    result = historical_evaluate_candidate(sequence, target)
    # ViennaRNA ensemble_defect is already normalized. Undo only the legacy
    # scorer's additional division, matching the paper candidate reaggregation.
    result['NED'] *= len(target)
    if not math.isfinite(result['NED']) or not 0 <= result['NED'] <= 1:
        raise ValueError('Invalid normalized ensemble defect')
    result['evaluation_valid'] = True
    result['generation_failed'] = False
    return result
