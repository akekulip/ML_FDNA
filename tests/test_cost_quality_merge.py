from pathlib import Path

import pytest

from scripts.p5_cost_quality_merge import sha256, verify_artifacts, verify_merge_sources


def test_relocated_bundle_verifies_local_bytes(tmp_path):
    traces = tmp_path / 'traces'
    traces.mkdir()
    artifacts = {}
    for name in ('traces.jsonl.gz', 'metrics.jsonl.gz'):
        path = traces / name
        path.write_bytes(b'saved data')
        artifacts[str(Path('/unavailable/original/location') / name)] = sha256(path)
    verify_artifacts(tmp_path, {'artifacts': artifacts})
    (traces / 'metrics.jsonl.gz').write_bytes(b'changed data')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_artifacts(tmp_path, {'artifacts': artifacts})


def test_merge_rejects_changed_live_summary_code(tmp_path):
    sources = {}
    for relative in ('scripts/p5_cost_quality.py', 'src/fdna/cost_quality.py'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen implementation')
        sources[str(Path('/old/checkout') / relative)] = sha256(path)
    verify_merge_sources(sources, tmp_path)
    (tmp_path / 'src/fdna/cost_quality.py').write_text('changed gate')
    with pytest.raises(ValueError, match='merge-time source differs'):
        verify_merge_sources(sources, tmp_path)
