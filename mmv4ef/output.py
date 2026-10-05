"""Keep private generated tables inside the ignored output directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_ignored_output(path):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT / 'outputs')
    except ValueError:
        raise ValueError('Generated outputs must be under the ignored outputs/ directory.') from None
    return resolved
