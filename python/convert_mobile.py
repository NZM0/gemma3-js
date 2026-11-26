#!/usr/bin/env python3
"""Convert an existing production FP32 export to W4A32, without downloading weights."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np


def convert(source, output):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('precision', 'fp32') != 'fp32' or manifest['head']['mode'] != 'full':
        raise ValueError('Expected full Reference FP32 export')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    output.mkdir(parents=True, exist_ok=True)

    def matrix(relative, shape):
        target = str(Path(relative).with_suffix('.q4'))
        src = source / relative
        if src.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError(f'Invalid size: {relative}')
        (output / target).parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as reader, (output / target).open('wb') as writer:
            while True:
                values = np.fromfile(reader, dtype='<f4', count=1024 * 1024)
                if not values.size:
                    break
                if not np.isfinite(values).all() or values.size % 32:
                    raise ValueError(f'Invalid INT4 weight: {relative}')
                groups = values.reshape(-1, 32)
                scale = np.max(np.abs(groups), axis=1) / 7
                scale[scale == 0] = 1
                q = (np.clip(np.rint(groups / scale[:, None]), -7, 7) + 8).astype('<u4')
                packed = np.zeros((len(groups), 5), dtype='<u4')
                packed[:, 0] = scale.astype('<f4').view('<u4')
                for lane in range(8):
                    packed[:, 1:] |= q[:, lane::8] << (4 * lane)
                packed.tofile(writer)
        return target

    head = manifest['head']
    head['file'] = matrix(head['file'], head['shape'])
    manifest['tokenEmbedding'] = head['file']
    # Fixed-size pieces permit bundled loading even on servers without Range support.
    row_bytes = head['shape'][1] * 20 // 32
    chunk_bytes = 8192 * row_bytes
    with (output / head['file']).open('rb') as reader:
        index = 0
        while data := reader.read(chunk_bytes):
            (output / f'head-part-{index:03d}.q4').write_bytes(data)
            index += 1
    head['chunkBytes'] = chunk_bytes
    for layer in manifest['layers']:
        directory = layer['directory']
        (output / directory).mkdir(exist_ok=True)
        for name, entry in layer['tensors'].items():
            relative = f"{directory}/{entry['file']}"
            if name.endswith('Proj'):
                entry['file'] = Path(matrix(relative, entry['shape'])).name
            else:
                shutil.copyfile(source / relative, output / relative)
    for name in [manifest['finalNorm'], 'chat-tokenizer.json', 'PROVENANCE.json']:
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    manifest['precision'] = 'w4a32'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=4) + '\n')
    # Only the chunked representation is needed by the Performance loader.
    (output / head['file']).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)
