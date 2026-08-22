---
date: 2026-08-22
datetime: 2026-08-22 16:34 JST
model: GPT-5 Codex
status: reference
topic: C-mVMC v1.3 official documentation snapshot
summary: |
  C-mVMC v1.3.0 のalgorithm、expert-mode入力仕様、出力仕様を、
  英語・日本語のupstream manual sourceから変更せず抽出したスナップショット。
---

# C-mVMC v1.3 official documentation snapshot

This directory contains selected, unmodified manual sources extracted from:

- repository: `issp-center-dev/mVMC`
- tag: `v1.3.0`
- commit: `d73d06bd529d3b2573f38eb5817c4a5f52971006`
- commit date: `2024-10-05T10:02:12+09:00`

Extraction commands:

```bash
git show v1.3.0:doc/en/source/algorithm.rst
git show v1.3.0:doc/ja/source/algorithm.rst
git show v1.3.0:doc/en/source/expert.rst
git show v1.3.0:doc/ja/source/expert.rst
git show v1.3.0:doc/en/source/output.rst
git show v1.3.0:doc/ja/source/output.rst
```

The `algorithm` documents are the baseline mathematical specification for wave
functions, stochastic reconfiguration, RBM, and Lanczos calculations. They are
not a v1.2-to-v1.3 delta. The `expert` documents define the expert-mode input
contract, including the General RBM blocks added for v1.3. The `output`
documents define the output-file contract used by parity fixtures. English and
Japanese variants are retained so that terminology can be checked against both
upstream versions.

These files remain subject to the upstream mVMC license. See the repository
license and `THIRD_PARTY_LICENSES.md` before redistributing them separately.
