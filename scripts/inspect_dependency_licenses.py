"""Record installed direct-dependency license evidence, not legal clearance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    requirements = Path(__file__).resolve().parents[1] / 'requirements.txt'
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, required = line.strip().split('==')
        distribution = importlib.metadata.distribution(name)
        if distribution.version.split('+')[0] != required:
            raise ValueError('Installed version differs from pin: ' + name)
        files = []
        for member in distribution.files or []:
            if 'license' not in str(member).lower() and 'copying' not in str(member).lower():
                continue
            path = distribution.locate_file(member)
            if path.is_file():
                files.append({'distribution_relative_path': str(member),
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if not files:
            raise ValueError('No installed license evidence: ' + name)
        rows.append({'package': name, 'required_version': required,
                     'installed_version': distribution.version,
                     'license_expression': distribution.metadata.get('License-Expression'),
                     'license_classifiers': [x for x in distribution.metadata.get_all('Classifier', [])
                                             if x.startswith('License ::')],
                     'license_files': files})
    result = {'scope': 'Installed direct dependencies only; excludes transitive, dataset, upstream model and project rights clearance',
              'license_audit_status': 'INCOMPLETE', 'dependencies': rows}
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print('RECORDED', len(rows), 'direct dependencies; not license PASS')


if __name__ == '__main__':
    main()
