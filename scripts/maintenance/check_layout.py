#!/usr/bin/env python3
"""Check repository files without importing navigation code or writing outputs."""
import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    names = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '--cached', '--others',
         '--exclude-standard', '-z'], text=True,
    ).split('\0')
    counts = {'python': 0, 'shell': 0, 'json': 0, 'yaml': 0, 'local_links': 0}
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
            counts['python'] += 1
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            counts['shell'] += 1
        elif path.suffix == '.json':
            json.loads(path.read_text())
            counts['json'] += 1
        elif path.suffix in {'.yaml', '.yml'}:
            yaml.safe_load(path.read_text())
            counts['yaml'] += 1
        elif path.suffix == '.md' and not name.startswith('docs/archive/'):
            body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            for match in re.finditer(r'\[[^\]]+\]\((<[^>]+>|[^\s)]+)\)', body):
                target = match.group(1).strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                destination = Path(unquote(parsed.path))
                if not destination.is_absolute():
                    destination = path.parent / destination
                if not destination.exists():
                    raise FileNotFoundError(f'{name}: {target}')
                counts['local_links'] += 1
    print(json.dumps({'status': 'PASS', 'counts': counts,
                      'scope': 'syntax, config parsing, current local Markdown file links',
                      'not_checked': ['historical archive links', 'remote URLs',
                                      'Markdown heading anchors', 'GPU/runtime behavior']}, indent=2))


if __name__ == '__main__':
    main()
