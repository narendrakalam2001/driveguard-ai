"""Environment sanity checks: catch broken installs at test time instead of in the middle of a video run."""

import pytest


def test_deepsort_can_be_imported():
    """deep-sort-realtime needs pkg_resources (setuptools<81); without it every video fails at tracker start-up."""
    pytest.importorskip("deep_sort_realtime")
    from deep_sort_realtime.deepsort_tracker import DeepSort  # noqa: F401
