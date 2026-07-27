"""Compatibility shim for tools that still invoke ``setup.py`` directly.

All package metadata and dependencies live in ``pyproject.toml``.
"""

from setuptools import setup


setup()
