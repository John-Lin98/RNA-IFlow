"""A shape-compatible but behavior-changing config must not load as U2442."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from load_export import parse_config


class ExportIdentityTest(unittest.TestCase):
    def test_pinned_config_and_activation_tampering(self):
        raw = (ROOT / 'configs/u2442.json').read_bytes()
        config = parse_config(raw)
        config['backbone_config']['hidden_act'] = 'relu'
        with self.assertRaisesRegex(ValueError, 'config SHA256 mismatch'):
            parse_config(json.dumps(config).encode())
        with self.assertRaisesRegex(ValueError, 'config SHA256 mismatch'):
            parse_config(raw + b' ')


if __name__ == '__main__':
    unittest.main()
