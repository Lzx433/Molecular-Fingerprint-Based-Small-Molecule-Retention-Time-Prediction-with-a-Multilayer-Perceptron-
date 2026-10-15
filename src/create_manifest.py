"""Create or verify SHA-256 checksums of the delivered research artifacts.

Portable dependency caches are excluded: requirements.txt and the recorded
environment describe these packages. The manifest itself is also excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    target = root / 'file_manifest.json'
    excluded = {'.runtime', '.torch_runtime', '__pycache__', '.venv'}
    if args.verify:
        manifest = json.loads(target.read_text(encoding='utf-8'))
        failures = []
        for record in manifest['files']:
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                failures.append(record['path'] + ': missing or invalid path')
            elif path.stat().st_size != record['bytes'] or checksum(path) != record['sha256']:
                failures.append(record['path'] + ': hash or size mismatch')
        if failures:
            raise SystemExit('\n'.join(failures))
        print(json.dumps({'status': 'pass', 'verified_files': len(manifest['files']),
                          'verified_bytes': sum(x['bytes'] for x in manifest['files'])}))
        return
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == target:
            continue
        relative = path.relative_to(root)
        if excluded.intersection(relative.parts):
            continue
        files.append({'path': relative.as_posix(), 'bytes': path.stat().st_size,
                      'sha256': checksum(path)})
    manifest = {'algorithm': 'SHA-256', 'excluded_directories': sorted(excluded), 'files': files}
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'created', 'files': len(files),
                      'bytes': sum(x['bytes'] for x in files)}))


if __name__ == '__main__':
    main()
