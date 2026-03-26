# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**asr-polish** is a unified ASR (Automatic Speech Recognition) framework for Chinese, focused on solving common production/commercial deployment issues with open-source ASR models (e.g., prompt tuning, streaming support gaps). Licensed under MIT.

Two main use-case categories:

1. **Real-time Subtitles** — WebSocket-based standard business protocol layer that abstracts underlying ASR model implementations. Models that don't support streaming are adapted at the engineering layer.
2. **Offline Audio File Recognition** — Batch processing of audio files.

## Current State

This project is in its initial planning phase. No source code, build system, tests, or dependencies have been established yet. The repository contains only README.md and LICENSE.
