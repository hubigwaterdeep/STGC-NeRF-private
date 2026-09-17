"""Select the explicitly audited local source; never edit or promote it."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/logs/intensity_fit_diagnosis_20260916/source'


def activate_source():
    required = ('best_core/scene_field.py', 'best_core/lidar4d.py', 'model/stgc_best.py')
    for relative in required:
        if not (SOURCE / relative).is_file():
            raise FileNotFoundError(f'Audited source is unavailable: {SOURCE / relative}')
    for name in ('best_core', 'model', 'data', 'utils', 'main_ours'):
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        paths = list(getattr(loaded, '__path__', []))
        if getattr(loaded, '__file__', None):
            paths.append(loaded.__file__)
        if any(not Path(p).resolve().is_relative_to(SOURCE) for p in paths):
            raise RuntimeError(f'{name} already resolves outside audited source; use a fresh process')
    sys.path[:0] = [str(SOURCE), str(SOURCE / '.deps')]
    return SOURCE
