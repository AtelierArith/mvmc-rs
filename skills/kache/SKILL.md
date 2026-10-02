---
name: kache
description: Install or update the kache Rust compiler cache, configure Cargo to use it, and verify cache hits. Use for requests to enable, upgrade, or troubleshoot kache.
---

Configure kache for the requested Cargo workspace and verify that the configured binary serves builds. Preserve existing compiler wrappers, Cargo settings and toolchain choices unless replacing them is part of the request.

## Install or update

Inspect `rustc --version`, `cargo --version`, the existing kache binary/version, applicable Cargo configuration, and `RUSTC_WRAPPER` / `RUSTC_WORKSPACE_WRAPPER` overrides. For current installation options, use the [official repository](https://github.com/kunobi-ninja/kache) and [configuration reference](https://github.com/kunobi-ninja/kache/blob/main/docs/getting-started/configuration.mdx).

Run the requested `cargo install kache`, normally with `--locked`. A release installation can take several minutes; poll the active process rather than launching duplicate installs. Confirm the installed binary's version and exact path.

When the global Cargo installation directory is unwritable, install with `--root` in a persistent writable location. An isolated writable `CARGO_HOME` may also be necessary for registry and installation metadata. Do not leave Cargo dependent on a binary stored only in a temporary directory. A workspace-local `.cargo/kache/` prefix is an option; ignore its binaries and machine-specific configuration in Git. Report a local installation accurately, without claiming that the global command was upgraded.

## Configure Cargo

Merge a `[build] rustc-wrapper` entry into the applicable `.cargo/config.toml`; do not overwrite unrelated tables. Use the installed binary's exact path when it is not on PATH. Resolve existing wrapper environment overrides before claiming that Cargo uses this configuration. Configure every active checkout in scope; avoid changing other projects or shell startup files unnecessarily.

For a workspace-local setup, this shape keeps the binary and configuration persistent:

```toml
[build]
rustc-wrapper = "/absolute/workspace/.cargo/kache/bin/kache"

[env]
KACHE_CONFIG = { value = "/absolute/workspace/.cargo/kache/config.toml", force = true }
```

Choose cache and runtime directories that are writable in the actual execution environment. Unless remote caching or automatic cleanup is requested, a local configuration can use:

```toml
[cache]
local_store = "/absolute/workspace/target/kache-cache"
runtime_dir = "/writable/runtime/mvmc-kache"
local_only = true
auto_clean_orphaned_targets = false
auto_clean_unused_units_days = 0
```

Check these keys against the installed version's documentation. Keep machine-specific absolute paths and binaries out of project commits. Cargo's `[env]` settings apply to subprocesses, not to standalone `kache` commands: pass the same `KACHE_CONFIG` explicitly when inspecting statistics.

## Verify observable cache reuse

Build a small dependency-free library twice with `CARGO_INCREMENTAL=0` and distinct disposable `--target-dir` directories, invoking Cargo from the configured workspace. Keep source paths, toolchain, profile and build flags identical. Distinct output directories ensure the second invocation actually reaches the wrapper instead of merely reusing Cargo's existing artifacts. Do not delete the user's target directory to force this check.

Use the configured binary's `stats --json` with the same `KACHE_CONFIG`. Compare statistics before and after, or use a fresh dedicated cache, to prove that the second build adds a local hit. A faster build or a populated cache alone is insufficient evidence. Also run a relevant Cargo command in the actual workspace to verify configuration discovery; do not substitute the smoke library for repository correctness checks.

Report the installed version, installation scope, configuration file, and verified hit result. If a toolchain update interrupts a live build, wait for its terminal result and restart required validation with the new compiler; a cache check does not replace that validation.
