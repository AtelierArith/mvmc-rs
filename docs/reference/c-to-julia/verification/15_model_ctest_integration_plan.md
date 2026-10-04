# Plan: add C-mVMC ctest-equivalent integration coverage

## Goal

Add public Julia-mVMC integration tests for the 15 non-MPI, non-UHF C-mVMC
ctest fixtures using **the same acceptance criteria as C-mVMC's ctest**.

This is intentionally weaker and more realistic than the private development
test that compared `zvo_out` step-by-step at `1e-10`. The public CI should
answer: "Would this model pass the original C-mVMC ctest-style statistical
criterion when run through Julia-mVMC?"

The source of truth is the original C-mVMC ctest fixture set: bring those
fixture inputs and reference statistics into Julia-mVMC, run the corresponding
Julia path, and apply the same pass/fail rule. The final target is all 15
models below; any temporary skip is only a staged implementation detail.

Staging should be explicit:

- Milestone 1: enable the 12 standard real/cmp/fsz fixtures.
- Milestone 2: enable `GeneralRBM_cmp` after RBM-bearing `initial.def`
  support lands in production code.
- Milestone 3: enable the two Lanczos fixtures after a real mode1/Lanczos
  driver and output path exist.

## Target models

| Group | Models |
|-------|--------|
| standard real | `HeisenbergChain`, `HubbardChain`, `HubbardTetragonal`, `HubbardTetragonal_MomentumProjection`, `KondoChain` |
| standard cmp | `HeisenbergChain_cmp`, `HubbardChain_cmp`, `KondoChain_cmp`, `KondoChain_Stot1_cmp` |
| standard fsz | `HeisenbergChain_fsz`, `HubbardChain_fsz`, `KondoChain_fsz` |
| expert | `GeneralRBM_cmp` |
| mode1 / Lanczos | `SpinChainLanczos`, `HubbardChainLanczos` |

Do not include `_mpi` or UHF tests in this pass. Public Julia-mVMC is
single-process, and UHF uses C-mVMC's `ComplexUHF` path rather than the VMC
parameter-optimization path.

## C-mVMC ctest criteria

C-mVMC's Python runners do not compare every `zvo_out` SR step. They compare
small output slices against committed reference means and standard deviations.

Important compatibility note: C-mVMC's `output/zqp_opt.dat` and the current
Julia-mVMC `output/zqp_opt.dat` are **not the same format**. In C, with
`NSROptItrSmp > 1` (the usual fixture setting), the first two loaded values
are `mean(<H>).real` and `mean(<H>).imag` from the averaged `OutputOptData`
entry used by ctest. With `NSROptItrSmp == 1`, the second value is a
hard-coded `0.0`. In current Julia-mVMC, `zqp_opt.dat` starts with
variational parameters in a multi-line `real imag` layout. Therefore a
ctest-equivalent runner must not parse Julia's current `zqp_opt.dat` and
compare its first two numbers to `ref_mean.dat`; that would compare different
physical quantities.

### Standard VMC runner

Used by `runtest.py` for real/cmp/fsz standard-mode fixtures.

1. Run:

   ```bash
   vmc.out -s data/<Model>/StdFace.def data/<Model>/initial.def
   ```

2. Read:

   ```text
   work/<Model>/output/zqp_opt.dat
   data/<Model>/ref/ref_mean.dat
   data/<Model>/ref/ref_std.dat
   ```

3. Compare the first two values from `zqp_opt.dat` against the first two
   reference values. C fails only when both conditions are true:

   ```text
   abs(calc - ref_mean) >= 3 * ref_std
   abs(calc - ref_mean) >= 1e-8
   ```

So a value passes when it is within the 3-sigma reference interval or within
the absolute floor `1e-8`.

### Expert runner

Used by `runtest_expert.py` for `GeneralRBM_cmp`.

The criterion is the same as the standard VMC runner, but C runs expert mode:

```bash
vmc.out -e namelist.def initial.def
```

and compares `output/zqp_opt.dat` against `ref/ref_mean.dat` and
`ref/ref_std.dat`.

### Mode1 / Lanczos runner

Used by `runtest_mode1.py` for `SpinChainLanczos` and
`HubbardChainLanczos`.

1. Run:

   ```bash
   vmc.out -s data/<Model>/StdFace.def data/<Model>/zqp_opt.dat
   ```

2. Compare the first two values from:

   ```text
   output/zvo_ls_out_001.dat
   ```

   against:

   ```text
   ref/ref_mean_Els.dat
   ref/ref_std_Els.dat
   ```

using the same `3 * std` plus `1e-8` floor criterion.

Julia-mVMC currently documents full Lanczos as not ported and does not expose
a C-compatible `zvo_ls_out_001.dat` output path. Treat the two Lanczos models
as a separate v0.2-sized phase: they need a mode1 driver, Lanczos MC
estimator, diagonalization, and C-compatible output before they can be called
ctest-equivalent.

## Public fixture layout

Use the existing public fixture style, but store C ctest reference statistics
rather than only `zvo_out_first10.dat`.

```text
test/integration/reference/<fixture_name>/
├── inputs/
│   ├── namelist.def
│   ├── *.def
│   └── initial.def or zqp_opt.dat when required by the C runner
└── ctest_ref/
    ├── ref_mean.dat
    └── ref_std.dat
```

For Lanczos fixtures:

```text
ctest_ref/
├── ref_mean_Els.dat
└── ref_std_Els.dat
```

`zvo_out_first10.dat` may remain as an optional diagnostic fixture for the
four existing bit-level checks, but it should not be the main acceptance
criterion for the 15-model public CI path.

## Implementation phases

The phases below are in logical order. The practical implementation order is
different: first add the required `run_para_opt_from_namelist` API support
(`nsmp` override plus C-compatible ctest comparison values), then generate
fixtures, then wire the ctest-equivalent runner, then handle Lanczos as a
separate follow-up, and finally update user-facing docs.

### Phase 1: split strict and ctest-equivalent tests

Keep the existing four-model `zvo_out_first10.dat` comparison as a strict
regression test if desired.

Add a new ctest-equivalent runner, for example:

```text
test/integration/ctest_equivalent.jl
```

The runner should use a model table with:

- fixture directory
- C model name
- runner kind: `:standard`, `:expert`, or `:lanczos`
- Julia mode label: `:real`, `:cmp`, or `:fsz`
- comparison source: C-compatible `OutputOptData` values returned by Julia,
  C-compatible `zqp_opt.dat` once implemented, or `zvo_ls_out_001.dat` for
  Lanczos
- reference files to load
- number of leading values to compare, normally `2`
- `initial_path`: explicit relative path such as `inputs/initial.def`,
  `inputs/zqp_opt.dat`, or `nothing`
- `nsteps_override`: `nothing` means use `modpara.def`; an integer means
  deliberately shorten the run
- `nsmp_override`: `nothing` means use `modpara.def`; an integer overrides
  `NSROptItrSmp`
- `expected_runtime_seconds`: CI budget estimate for the fixture
- `blocked_by`: optional implementation gate, for example
  `RBM_INITIAL_DEF` or `LANCZOS_MODE1`

The pass/fail predicate should mirror C exactly:

```julia
abs(calc - mean) < 3 * std || abs(calc - mean) < 1e-8
```

Use a strict `>=` failure condition if implementing it in negative form, to
match C's Python runner.

### Phase 2: generate and bundle fixtures

Add a helper such as:

```text
test/integration/tools/generate_ctest_fixtures.jl
```

The helper should take a C-mVMC source/build path and copy:

- generated expert-mode input files from `build/test/python/work/<Model>/`
  into `inputs/` for standard and mode1 fixtures
- checked-in expert-mode input files from `test/python/data/<Model>/*.def`
  into `inputs/` for expert fixtures such as `GeneralRBM_cmp`
- `initial.def` or `zqp_opt.dat` from `test/python/data/<Model>/` when the
  C runner uses them
- C reference statistics from `test/python/data/<Model>/ref/` into
  `ctest_ref/`

For standard-mode fixtures, C starts from `StdFace.def` and generates expert
files in `work/<Model>/`; Julia-mVMC should run from the generated
`namelist.def` and `.def` files so the public test does not require
`mvmc_dry.out`.

The helper must fail on missing required files. Do not silently create a
partial fixture.

### Phase 3: make Julia output comparable to ctest

The current public Julia output is not directly comparable to C ctest because
Julia's `zqp_opt.dat` does not contain C's `OutputOptData` leading entries.
Before enabling the non-Lanczos fixtures, implement one of these two
paths:

1. Preferred first step: extend `run_para_opt_from_namelist` (or a new
   internal result type) to return the C ctest comparison values directly,
   for example `ctest_values::Vector{Float64}` with the first two entries
   corresponding to C's `np.loadtxt("output/zqp_opt.dat")[0:2]`.
2. Later compatibility step: implement a C-compatible `OutputOptData` /
   `zqp_opt.dat` writer in Julia and have the runner parse that file.

Do not use current Julia `output/zqp_opt.dat` as the comparison source until
the C-compatible writer exists.

Required API work:

- Add `nsmp::Union{Integer,Nothing} = nothing` to
  `run_para_opt_from_namelist`. When provided, it must override
  `data.modpara.nsr_opt_itr_smp` before `vmc_para_opt!` runs. `nsteps` can
  stay explicit, with the ctest runner pre-parsing `modpara.def` to supply
  the default value when `nsteps_override === nothing`.
- Add a C-compatible comparison channel, such as
  `result.ctest_values::Vector{Float64}`, or implement a separate
  C-compatible `OutputOptData` writer. The existing Julia `zqp_opt.dat`
  must not be used for ctest comparison until this writer exists.
- Assert `effective_nsteps >= effective_nsmp` in the runner or wrapper.
  The current `store_opt_data!` / `vmc_para_opt!` buffer behavior is not
  valid for `nsteps < nsmp`; it leaves leading sample slots as zero-valued
  placeholders and would pull the average toward zero.

For the 12 standard real/cmp/fsz models, the intended public path is:

```julia
result = MVMCOptimizers.run_para_opt_from_namelist(...)
calc = result.ctest_values[1:2]  # or parse C-compatible OutputOptData output
```

Then compare `calc` to the first two values in `ctest_ref/ref_mean.dat` and
`ctest_ref/ref_std.dat`.

The run length must also match the C fixture semantics. C references are
generated after the fixture's full `NSROptItrStep` run and average over
`NSROptItrSmp`; the existing strict public test's `nsteps=10` shortcut is
not valid for ctest-equivalent checks. The default must be:

```julia
nsteps = something(model.nsteps_override, parsed_modpara.nsr_opt_itr_step)
nsmp   = something(model.nsmp_override, parsed_modpara.nsr_opt_itr_smp)
```

Use overrides only when the model table explicitly documents the shortened
run and its effect on the reference criterion.

Before enabling the 12 standard fixtures, run a preflight matrix:

- confirm current public `run_para_opt_from_namelist` can reproduce the C
  starting state
- check for `OptTrans` / `InOptTrans`; initial inspection of the 15 source
  fixtures did not show these, so treat OptTrans as a gate rather than
  required scope

`GeneralRBM_cmp` is a confirmed additional gate, not a speculative one:
current production `read_initial_def!` refuses RBM-bearing `initial.def`
files. Add `blocked_by = "RBM_INITIAL_DEF"` in the model table until
`read_initial_def!` consumes the NRBM triples in the C order
Proj -> RBM -> Slater -> OptTrans.

Do not reimplement initial-parameter loading inside the test runner. The
ctest-equivalent runner must use the public production path
(`run_para_opt_from_namelist`, which internally calls `read_initial_def!`) so
loader behavior is shared with user code.

### Phase 4: handle Lanczos separately

C-mVMC's Lanczos ctest checks `zvo_ls_out_001.dat`, not `zqp_opt.dat` and not
`zvo_out.dat`.

This is not just a missing output file. The C fixtures use `NVMCCalMode = 1`
and `NLanczosMode = 1`, which means they run through C's `VMCPhysCal` /
mode1 path. Julia's `run_para_opt_from_namelist` always calls
`vmc_para_opt!`, and there is no C-compatible Lanczos mode1 implementation
or `zvo_ls_out_001.dat` output today.

Minimum required work for the two Lanczos fixtures:

1. Add a driver entry such as `run_phys_cal_from_namelist`, or dispatch on
   `NVMCCalMode` before choosing `vmc_para_opt!` vs `vmc_phys_cal!`.
2. Implement the Lanczos branch in `vmc_phys_cal!`, including the MC
   estimates needed for the small Lanczos matrix (`<H>`, `<H^2>`, `<H^3>`
   or the C-equivalent quantities).
3. Add the small-matrix diagonalization and C-compatible
   `zvo_ls_out_001.dat` values.

Treat this as a v0.2-sized follow-up unless the above work is explicitly
in scope. Until then, include the two Lanczos fixture entries with
`blocked_by = "LANCZOS_MODE1"` and document them as skipped.

Do not call a step-0 `zvo_out` comparison "ctest-equivalent"; it is useful as
a development check, but it is not the original C ctest criterion.

### Phase 5: documentation updates

Update:

- `test/integration/reference/README.md`
  - list the ctest-equivalent fixtures and their C runner kind
  - document `ref_mean` / `ref_std` provenance
  - document skipped/pending Lanczos status if mode1 is not implemented yet
- `docs/manual/03_optimization.md`
  - distinguish strict `zvo_out_first10` regression tests from
    ctest-equivalent statistical tests
- `docs/manual/05_compatibility.md`
  - report which models pass ctest-equivalent public CI
- top-level `README.md`
  - update the verified status once the ctest-equivalent suite is enabled

## Acceptance criteria

- The public ctest-equivalent runner passes for all enabled non-MPI, non-UHF
  fixtures using the same `3 * std` plus `1e-8` criterion as C-mVMC.
- The first enabled milestone should pass the 12 standard real/cmp/fsz
  fixtures.
- `GeneralRBM_cmp` should remain blocked until RBM-bearing `initial.def`
  support is implemented in production `read_initial_def!`.
- The two Lanczos fixtures should remain blocked until a real mode1/Lanczos
  driver and `zvo_ls_out_001.dat`-equivalent comparison path exist.
- The model table records `nsteps_override`, `nsmp_override`, and an
  `expected_runtime_seconds` budget for every fixture. CI should have an
  explicit total runtime target before all 12 standard fixtures are
  enabled.
- The runner asserts `effective_nsteps >= effective_nsmp`.
- Fixture generation is reproducible from a C-mVMC checkout/build and leaves
  no dependency on local `private-mVMC/...` paths.
- Required fixture files are checked eagerly; missing refs or inputs are test
  failures, not silent skips.
- Note: for fixtures where `ref_std` is smaller than the `1e-8` absolute
  floor, the ctest criterion is effectively `abs(calc - ref_mean) < 1e-8`.
  This is still the C-mVMC criterion, but it is not a loose statistical test.

## Risks and checks

- **Too-strong comparison by accident**: do not use first-10-step `zvo_out`
  equality as the acceptance criterion for the 15-model public suite.
- **Wrong `zqp_opt.dat` semantics**: current Julia `zqp_opt.dat` starts with
  variational parameters, while C ctest reads energy-like `OutputOptData`
  values. Implement `result.ctest_values` or a C-compatible writer before
  comparing.
- **Incomplete run length**: do not reuse the strict test's `nsteps=10`
  shortcut for ctest-equivalent checks. C reference statistics are generated
  from full fixture runs and `NSROptItrSmp` averaging.
- **Override invariant**: `effective_nsteps` must be greater than or equal to
  `effective_nsmp`; otherwise the current optimization sample buffer can
  include zero-valued placeholder samples in the average.
- **Initial-state drift**: if `initial.def` is absent, match C behavior and
  fall back to random initialization. If `initial.def` is present but cannot
  be loaded, fail or mark the model unsupported; do not silently fall back to
  random initial parameters.
- **RBM initial parameters**: `GeneralRBM_cmp` requires extending production
  `read_initial_def!` to consume the RBM block safely.
- **Regenerated `initial.def` files**: fixtures such as
  `HubbardTetragonal_MomentumProjection` have checked-in `initial.def` files
  produced by C-mVMC helper scripts such as `AddRand.py` using Python's RNG.
  Do not regenerate those perturbations with Julia's RNG and expect identical
  references.
- **OptTrans**: no `OptTrans` / `InOptTrans` was found in the inspected 15
  source fixtures, but regenerate and scan fixtures before enabling the full
  set.
- **NSplitSize**: public Julia-mVMC is single-process. If regenerated
  fixtures accidentally come from `_mpi` variants or rely on split-process
  behavior, keep them out of this suite.
- **Generated-path leakage**: scan committed fixtures:

  ```bash
  rg '/Users|Dropbox|private-mVMC|build/test/python/work' test/integration/reference
  ```
- **OptTrans scan**: after fixture generation, run:

  ```bash
  rg -l 'OptTrans|InOptTrans' test/integration/reference/*/inputs
  ```
