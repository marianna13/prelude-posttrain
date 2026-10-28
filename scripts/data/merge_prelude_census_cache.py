#!/usr/bin/env python3
"""Adopt complete census work units from an identical frozen build.

Publish metadata before completion markers. Hard links avoid duplicating data;
separate scratch directories let concurrent scans finish without temporary-file
collisions. Existing completed units must agree in their counts or checksums.
"""
import argparse
import json
import os
from pathlib import Path


def merge(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / 'input_manifest.json').read_text())
    target = json.loads((destination / 'input_manifest.json').read_text())
    if manifest != target:
        raise ValueError('Cache inventories differ')
    adopted = 0
    for marker in sorted((source / 'tasks').glob('*.json')):
        result = json.loads(marker.read_text())
        if result.get('limited_to_rows') or result['build_id'] != manifest['build_id']:
            continue
        completed = destination / 'tasks' / marker.name
        if completed.exists():
            previous = json.loads(completed.read_text())
            if not previous.get('limited_to_rows'):
                if previous['build_id'] != result['build_id']:
                    raise ValueError('Conflicting build: ' + marker.name)
                for field in ('counts', 'sha256'):
                    if previous.get(field) != result.get(field):
                        raise ValueError('Conflicting results: ' + marker.name)
                continue
        for directory in ('metrics', 'stack_files', 'ultra_files'):
            path = source / directory / (marker.stem + '.parquet')
            if not path.exists():
                continue
            final = destination / directory / path.name
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(final.name + f'.adopt-{os.getpid()}')
            os.link(path, temporary)
            os.replace(temporary, final)
        completed.parent.mkdir(exist_ok=True)
        temporary = completed.with_name(completed.name + f'.adopt-{os.getpid()}')
        os.link(marker, temporary)
        os.replace(temporary, completed)
        adopted += 1
    print(json.dumps(dict(adopted=adopted, source=str(source), destination=str(destination))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    merge(args.source, args.destination)
