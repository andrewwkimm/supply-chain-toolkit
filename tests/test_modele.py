"""Tests for sctk."""

from sctk import __version__


def test_version() -> None:
    """Tests that version is set to expected value."""
    assert __version__ == "0.1.0"
