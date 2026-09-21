"""Small independent ViennaRNA regression for the published scoring convention."""
from pathlib import Path
import sys
import unittest

import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper_metrics import evaluate_candidate


class PaperMetricsTest(unittest.TestCase):
    def test_normalized_defect_and_probability(self):
        for sequence, target in [('GCCAAAGGC', '(((...)))'), ('AAAAAAAAA', '(((...)))')]:
            with self.subTest(sequence=sequence):
                md = RNA.md()
                md.temperature, md.dangles, md.uniq_ML = 37., 2, 1
                compound = RNA.fold_compound(sequence, md)
                _, energy = compound.mfe()
                structures = {r.structure for r in compound.subopt(0)}
                compound.exp_params_rescale(energy)
                compound.pf()
                result = evaluate_candidate(sequence, target)
                self.assertAlmostEqual(result['NED'], compound.ensemble_defect(target), places=12)
                self.assertAlmostEqual(result['target_probability'], compound.pr_structure(target), places=12)
                self.assertEqual(result['mfe_hit'], target in structures)
                self.assertEqual(result['uMFE_hit'], len(structures) == 1 and target in structures)

    def test_invalid_input(self):
        for sequence, target in [('', ''), ('AX', '..'), ('AAA', '..')]:
            with self.assertRaises(ValueError):
                evaluate_candidate(sequence, target)


if __name__ == '__main__':
    unittest.main()
