# Optional actual SFMT state/count observer

Developer-only `reviewed_sfmt_state.c` and `.jl` provide observation of the
reference process's **actual** raw SFMT state, index and primitive draw count.
No ordinary Rust build/test invokes them. They do not reconstruct an old
capture, replay a sampler, generate expected parameters, or change a numerical
parameter/solver function. Install only before a fresh original seed call.

## Original source and compilation boundary

The C driver includes the complete unchanged reference
`extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT.c` with only entry-symbol renaming
macros for `gen_rand32` and `init_gen_rand`. Their original arithmetic bodies
are unchanged. Exported wrappers call those original bodies once, count one
32-bit word or reset the count after the original seed. The read-only getter
copies `psfmt32[0:624]`, `idx` and count; it rejects use before initialization.

**`SFMT-real.c` is compiled as a separate translation unit**, outside those
renaming macros. Its unchanged `genrand_real2` calls the exported counted
`gen_rand32`. Conversely, `sfmt_dump_rand32` inside the included `SFMT.c`
calls the renamed original primitive, saving/restoring the actual state/index,
and never changes the observed draw count. Including SFMT-real.c inside the
renamed unit would violate this contract; that is not the executed build.

Upstream whole-source hashes:

| Input | SHA256 |
| --- | --- |
| SFMT.c | 4eed94fb587cefa3022d259379fdd492ddd76a4709d23aae9c7dd6ebc52db321 |
| SFMT-real.c | 1e283c215c48ec94a1083c963314bd4a1c6d4498fe4ffbbbb0b80240178de36e |
| SFMT.h | 8573dbde79f1da2d2d7620d8bf696f93c965b3bd5684b9ffd14bea11f2b229d1 |
| SFMT-params.h | 99374c27864918b6babacd9b7db10767ff855892065522b41a1bf32f5c263154 |
| SFMT-params19937.h | 42a605c202dc146095b3ba8b47f9275039c4e17bf039eb567c4985c2a662dc35 |

Original SFMT copyright/BSD notices remain in the included source; reference
package license context is `SFMT.jl/THIRD_PARTY_LICENSES.md` and `LICENSE`.
No vendored file is edited. Baseline compilation defines
`REVIEWED_SFMT_BASELINE`, retaining original symbols/bodies plus the same
read-only state getter; baseline count is deliberately zero, not a claim
about baseline draw consumption. Both libraries are loaded local/deep-bound
to prevent symbol interposition between independent test states.

Actual Linux x86_64/GCC13.3.0 commands (no BLAS):

```sh
gcc -std=gnu11 -O3 -shared -fPIC -DMEXP=19937 -I/workspaces/mvmc-rs/extern/Julia-mVMC/SFMT.jl/deps/sfmt /workspaces/mvmc-rs/c_toolbox/reviewed_sfmt_state.c /workspaces/mvmc-rs/extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT-real.c -o /home/vscode/.cache/mvmc/reviewed-c-parameter.AFmMKn/libsfmt-observed-final.so
gcc -std=gnu11 -O3 -shared -fPIC -DMEXP=19937 -DREVIEWED_SFMT_BASELINE -I/workspaces/mvmc-rs/extern/Julia-mVMC/SFMT.jl/deps/sfmt /workspaces/mvmc-rs/c_toolbox/reviewed_sfmt_state.c /workspaces/mvmc-rs/extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT-real.c -o /home/vscode/.cache/mvmc/reviewed-c-parameter.AFmMKn/libsfmt-baseline-final.so
```

Final C adapter SHA256
`8a8d8106a33a19271ff07375038d0702e2afbd80f051cc841e610f1fe4f86aeb`;
Julia API SHA256
`91102d67e9674c3bbc3fb1ceca8b4dbfa32c2440219a8b4e48e2a70313ec2591`;
checker SHA256
`c58769c6e01d7178375446b5260e4d7c416a1944b58cd138dce0bfc564478730`.
Actual observed library SHA256
`7e77954acae2073591edf17b5c9825f0c020a2525026ae450ac31b780d7a7c86`;
baseline SHA256
`a98385983cd7bea794ea4461cd46e1b763cf32b4a94260de84bee848edba1bb7`.

## API and actual verification

```julia
include("c_toolbox/reviewed_sfmt_state.jl")
ReviewedSFMTState.install!("EXACT_REVIEWED_OBSERVED_LIBRARY")
# Now the original generator seeds and initializes, unchanged.
ReviewedSFMTState.capture!("group-1-initialized-state.txt")
```

Install after any older C_API diagnostic overrides, but **before the original
seed**; do not install in an already-running capture or claim old-stage proof.
Capture at initialized, overlaid and synchronized boundaries for each group.
It reads the actual SFMT global, so no RNG replay/clone argument is needed.
The three lines match the independent C fixtures: 624 UInt32 state words,
actual index, observed UInt64 primitive count. `capture!` asserts that the
getter/write operation leaves the full snapshot unchanged. Unsupported
64-bit/bulk/real1/real3/res53/array-seed entry points fail closed.

Executed final on/off checker **36610**, terminal0 (completion chunk603fb8;
stdout chunk e9024d). Three seeds (1,12395,UInt32 maximum), 6,144 mixed
original integer/real2 draws. Exact raw state/index checked at every boundary,
observed count checked against actual calls, next624 equality and state/count
unchanged across peeks at block boundaries. Command:

```sh
julia +1.13.1 c_toolbox/check_reviewed_sfmt_state.jl BASELINE_LIBRARY OBSERVED_LIBRARY
```

Additional actual Julia SFMT API smoke, terminal0 chunk2c7196: original
package's 192 words and next624 match the observed API, index/count192,
getter/peek non-consuming and pre-seed getter rejects. This verifies routing
and observer invariance, **not the complete GeneralRBM initializer capture**.
Raw state/count comparisons across the fresh reviewed-fork 918 slot records
remain pending the owner installing these hooks in a new acquisition.
No model/CG trajectory acceptance, numerical tolerance or macOS result is
implied by the observer checks.
