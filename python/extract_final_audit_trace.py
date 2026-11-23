"""Extract a browser final-audit parity trace for Google's official oracle."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=Path('gemma3-boundary-trace.json'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report.get('schema') != 'gemma3-js-final-release-audit-v1':
        parser.error('Not a final-release audit report')
    trace = report.get('trace')
    if not trace:
        parser.error('No trace (browser generation failed)')
    from compare_oracle_json import validate_trace
    validate_trace(trace)
    if len(trace['turns'][0]['promptTokenIds']) <= 512:
        parser.error('Prompt did not cross 512 tokens')
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=4), encoding='utf-8')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
