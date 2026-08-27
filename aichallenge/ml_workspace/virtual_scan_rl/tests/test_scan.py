import numpy as np

from virtual_scan_rl.scan import ScanHistory, sanitize_scan


def test_sanitize_scan_replaces_invalid_and_resamples():
    result = sanitize_scan([0.0, np.nan, np.inf, 15.0], num_rays=8, max_range_m=30.0)
    assert result.shape == (8,)
    assert result.dtype == np.float32
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_scan_history_stacks_oldest_to_newest():
    history = ScanHistory(3, 4)
    history.reset(np.zeros(4, dtype=np.float32))
    value = history.append(np.ones(4, dtype=np.float32))
    assert value.shape == (3, 4)
    assert np.all(value[-1] == 1.0)

