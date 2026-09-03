# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""Croft — installer package.

The single source of truth for the project version. Everything that reports a
version (the CLI, the web app, pyproject.toml) reads it from here, so there is
exactly one line to bump at release time.
"""
__version__ = "0.1.0"
