"""Compatibility module for the historical ShapCRN entry point."""

from shapcrn.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
