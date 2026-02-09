# Agent Development Guide

A file for [guiding coding agents](https://agents.md/).

## General Principles

- Search related files for context before starting a task.
- Use `uv` for Python.
- After finishing a task, run `ruff` and `pyrefly` to check and fix code style and static issues.
- If a Codex MCP is available, invoke it to help verify code and validate ideas.