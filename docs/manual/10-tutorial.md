# 10. Tutorial: a 16-site Hubbard chain

[Contents](README.md) · Previous: [9. Output files](09-output-files.md) · Next: [11. Compatibility and differences](11-compatibility.md)

This tutorial optimizes the variational wave function of a half-filled one-dimensional Hubbard model and then measures
Green functions and the single-step Lanczos energy with the optimized parameters. Every command was run with a release build of
`mvmc` (commit `3e9024ee` of `main`, Linux x86_64, `cargo build --release -p mvmc-cli`); the excerpts below are real output
**(observed)**. Timings depend on the machine and its load (the optimization took 13 s and 31 s in two runs here).

## 10.1 Model and inputs

The inputs are committed in the repository, so no submodule or C tool is needed:
`benchmark/hubbard_chain/inputs/hubbard_chain_L16/` (generated with the C StdFace tool from the `StdFace.def` in the same directory; provenance in
`benchmark/hubbard_chain/README.md`). The model is

$$
\mathcal H=-t\sum_{i\sigma}\big(c^\dagger_{i\sigma}c_{i+1\sigma}+{\rm h.c.}\big)+U\sum_in_{i\uparrow}n_{i\downarrow},\qquad t=1,\ U=4,\ N_s=16,\ N=16\ \text{(half filling)} .
$$

```bash
mkdir work && cd work
cp -r <repo>/benchmark/hubbard_chain/inputs/hubbard_chain_L16 hubbard_L16
cat hubbard_L16/namelist.def
```

```text
         ModPara  modpara.def
         LocSpin  locspn.def
           Trans  trans.def
    CoulombIntra  coulombintra.def
        OneBodyG  greenone.def
        TwoBodyG  greentwo.def
      Gutzwiller  gutzwilleridx.def
         Jastrow  jastrowidx.def
         Orbital  orbitalidx.def
        TransSym  qptransidx.def
```

What the files define (see [chapter 7](07-input-files.md)):

| File | Content in this example |
|------|-------------------------|
| `trans.def` | `NTransfer 64`: nearest-neighbour hopping $t=1$ for both spins on the 16-site ring |
| `coulombintra.def` | `NCoulombIntra 16`, $U_i=4$ |
| `locspn.def` | `NlocalSpin 0`: all electrons itinerant |
| `gutzwilleridx.def` | `NGutzwillerIdx 4`: Gutzwiller parameters shared with period 4 (`Lsub = 4`) |
| `jastrowidx.def` | `NJastrowIdx 32`: Jastrow parameters by distance class |
| `orbitalidx.def` | `NOrbitalIdx 64`: pair orbitals $f_{ij}$ (period-4 sublattice classes), real |
| `qptransidx.def` | `NQPTrans 4`: translation patterns (only the first $\lvert N_{\rm MP}\rvert$ are used) |
| `greenone.def` | `NCisAjs 32`: $\langle c^\dagger_{0\sigma}c_{j\sigma}\rangle$, $j=0..15$, both spins |
| `greentwo.def` | `NCisAjsCktAltDC 96`: a set of two-body correlators |

and the settings of `modpara.def` (the C key `Ncond` gives $N_e=8$; `2Sz 0`, `NSPGaussLeg 8` and `NSPStot 0` give a singlet projection with 8 mesh
points; `NMPTrans -1` selects a single translation sector with anti-periodic boundary conditions):

```text
Nsite 16   Ncond 16   2Sz 0   NSPGaussLeg 8   NSPStot 0   NMPTrans -1
NSROptItrStep 300   NSROptItrSmp 30   DSROptRedCut 1e-8   DSROptStaDel 0.01   DSROptStepDt 0.003
NVMCWarmUp 10   NVMCInterval 1   NVMCSample 300   NExUpdatePath 0   RndSeed 1   NSplitSize 1   NStore 1   NSRCG 0
```

There are $N_{\rm para}=4+32+64=100$ real variational parameters (all index files have `ComplexType 0`, so the real kernels are used),
$N_{\rm QP}=N_{\rm GL}\lvert N_{\rm MP}\rvertN_{\rm opt}=8$ projection sectors and $N_e=8$ electrons per spin.

## 10.2 Optimize the parameters

```bash
mvmc hubbard_L16/namelist.def --out-dir out_opt
```

(with `mvmc` = `<repo>/target/release/mvmc`). Console output:

```text
model    : Nsite=16 Nelec=8 NSROptItrStep=300
mode     : NVMCCalMode=0
sample   : NVMCSample=300 NVMCWarmUp=10

=== mvmc — Julia-mVMC Rust port ===
namelist : hubbard_L16/namelist.def
out-dir  : out_opt


=== Completed 300 SR steps in 30.67s ===
Output files written to: /…/out_opt
Final energy / site: -0.5418807043
Final-window means (30 steps): [-8.607008448725692, 0.0]
```

The window mean of the energy is $-8.6070$, i.e. $-0.5379$ per site; the last single step ($-0.5419$) is not an average and fluctuates.
Files written (`ls -l`): `zvo_out.dat` (46 500 bytes, 300 rows), `zvo_var.dat` (1.7 MB, 300 rows of 306 numbers),
`zqp_opt.dat`, `zqp_gutzwiller_opt.dat`, `zqp_jastrow_opt.dat`, `zqp_orbital_opt.dat`. The direct SR solver writes no `zvo_SRinfo.dat`
([9.4](09-output-files.md#94-solver-information-zvo_srinfodat)).

### Convergence: `zvo_out.dat`

Columns are $\operatorname{Re}\langle H\rangle$, $\operatorname{Im}\langle H\rangle$, $\langle\lvert E_{\rm loc}\rvert^2\rangle$, relative variance, $\langle S_z\rangle$, $\langle S_z^2\rangle$
([9.1](09-output-files.md#91-energy-per-step-zvo_outdat-opt-and-zvo_out_nnndat-phys)). Selected rows:

| Step (1-based) | $\langle H\rangle$ | $\langle H\rangle/N_s$ | relative variance |
|----------------|--------------------|------------------------|-------------------|
| 1 | +17.1987 | +1.075 | 0.1005 |
| 100 | −4.5697 | −0.286 | 0.5347 |
| 200 | −8.3101 | −0.519 | 0.0253 |
| 300 | −8.6701 | −0.542 | 0.0261 |

The first row is the energy of the random initial state ($+1.07$ per site); SR drives it down and the relative variance
(zero for an eigenstate) falls from $10^{-1}$ to $2.6\times10^{-2}$. The row at step 100 is in the middle of the optimization, which is why it is unconverged.

### Optimized parameters: `zqp_opt.dat`

A single line of triples (mean real, mean imaginary, standard deviation over the last 30 steps) for $\langle H\rangle$, $\langle H^2\rangle$ and the 100 parameters:

```text
-8.607008448725691707e+00  0.000000000000000000e+00  1.152418507247792157e-01  7.590574786855339084e+01  0.000000000000000000e+00  1.892906971655880399e+00 -1.556663106607272473e+00 …
```

so the energy window mean is $-8.6070\pm0.115$ (standard deviation of the 30 step energies). `zqp_gutzwiller_opt.dat` holds the four Gutzwiller parameters:

```text
======================
NGutzwillerIdx  4
======================
======================
======================
0 -1.556663106607272473e+00  0.000000000000000000e+00 
1 -1.337680080798275606e+00  0.000000000000000000e+00 
2 -1.734280992033364166e+00  0.000000000000000000e+00 
3 -1.676363487003552644e+00  0.000000000000000000e+00 
```

## 10.3 Physical quantities with the optimized parameters

PhysCal needs `NVMCCalMode 1` in the `ModPara` file. Make a copy of the inputs, switch the mode and use more samples for a better estimate:

```bash
cp -r hubbard_L16 phys_in
sed -i 's/^NVMCCalMode    0/NVMCCalMode    1/; s/^NVMCSample     300/NVMCSample     3000/' phys_in/modpara.def
mvmc phys_in/namelist.def --physcal out_opt/zqp_opt.dat --out-dir out_phys
```

```text
model    : Nsite=16 Nelec=8 NSROptItrStep=300
mode     : NVMCCalMode=1
sample   : NVMCSample=3000 NVMCWarmUp=10

=== mvmc — Julia-mVMC Rust port ===
namelist : phys_in/namelist.def
out-dir  : out_phys
physcal  : out_opt/zqp_opt.dat


=== Completed 1 PhysCal samples in 1.41s ===
Output files written to: out_phys
```

One sample (`NDataQtySmp 1`, `NDataIdxStart 1`) produces the files `zvo_out_001.dat`, `zvo_var_001.dat`, `zvo_cisajs_001.dat` and `zvo_cisajscktalt_001.dat`
(no `TwoBodyGEx` was requested). The dispatch rule of [8.1](08-running.md#selecting-the-calculation) is enforced: running `hubbard_L16` (mode 0) with `--physcal`, or `phys_in` (mode 1) without it, stops immediately:

```text
error: --physcal requires NVMCCalMode=1 in ModPara (found NVMCCalMode=0); set NVMCCalMode=1 for fixed-parameter PhysCal
error: NVMCCalMode=1 selects fixed-parameter PhysCal; supply the fixed parameter file with --physcal <PATH>
```

`zvo_out_001.dat` contains the energy of the fixed parameters measured with 3000 samples:

```text
-8.604176648473060851e+00  0.000000000000000000e+00   7.579059132541173938e+01  2.375646954033361000e-02 0.000000000000000000e+00 0.000000000000000000e+00
```

and the first rows of `zvo_cisajs_001.dat` ($i\,\sigma\,j\,\sigma'$ then real and imaginary part of $\langle c^\dagger_{i\sigma}c_{j\sigma'}\rangle$) are

```text
0 0 0 0  4.156666666666666288e-01   0.000000000000000000e+00 
0 0 1 0  3.677613437662305418e-01   0.000000000000000000e+00 
0 0 2 0 -2.158749322082987102e-02   0.000000000000000000e+00 
0 0 3 0 -8.735104226616838274e-02   0.000000000000000000e+00 
```

The first row is the density $\langle n_{0\uparrow}\rangle$ and the following rows the equal-spin one-body correlations with $j=1,2,3$. In `zvo_cisajscktalt_001.dat`, `0 0 0 0 0 0 0 0` is
$\langle n_{0\uparrow}n_{0\uparrow}\rangle=\langle n_{0\uparrow}\rangle$ and `0 0 0 0 0 1 0 1` is $\langle n_{0\uparrow}n_{0\downarrow}\rangle$ (double occupancy, 0.102).

### Statistical error: use several seeds

A single chain has a statistical error much larger than the naive binomial estimate, because consecutive samples are correlated. Repeating the measurement with
different seeds (`--seed N`; each run takes about 1.4 s) gives, for the same parameters and 3000 samples **(observed)**:

| `--seed` | $\langle H\rangle$ | relative variance | $\langle n_{0\uparrow}\rangle$ |
|----------|--------------------|-------------------|-------------------------------|
| 1 (= `RndSeed`) | −8.6042 | 0.0238 | 0.4157 |
| 2 | −8.6305 | 0.0224 | 0.5247 |
| 3 | −8.6261 | 0.0239 | 0.4580 |
| 4 | −8.6483 | 0.0201 | 0.4690 |

The energy is reproducible at the $10^{-2}$ level while the local density varies by about $\pm0.05$ between chains (the Gutzwiller and orbital parameters are shared with period 4, so the density of this state need not be uniform); the average of the four densities, 0.467, is closer to the filling 0.5 than any single value suggests. For production work increase `NVMCSample`, use `NDataQtySmp > 1` (several numbered output sets) or several MPI ranks
([8.4](08-running.md#84-mpi-and-grouped-execution)). The seed `1` row is identical to the run above because `--seed 1` equals the seed in `modpara.def`.

## 10.4 Single-step Lanczos energy

Set `NLanczosMode 1` (this input has neither `InterAll` nor spin-flip hopping, so the Lanczos path is supported, [7.5](07-input-files.md#75-supported-and-rejected-inputs)):

```bash
cp -r phys_in phys_ls
sed -i 's/^NLanczosMode   0/NLanczosMode   1/' phys_ls/modpara.def
mvmc phys_ls/namelist.def --physcal out_opt/zqp_opt.dat --out-dir out_ls
```

This added `zvo_ls_out_001.dat` and `zvo_ls_qqqq_001.dat` (and took 23 s because the second moment $F(x,H^2)$ is evaluated from hopped configurations, [6.2](06-theory-observables-lanczos.md#62-the-single-step-lanczos-wave-function)).

```text
$ cat out_ls/zvo_ls_out_001.dat
-8.893351117533208949e+00   6.611265253057749952e-03   3.885060397405710741e-01
```

$E_{\rm LS}=-8.8934$ ($-0.5558$ per site), the relative variance dropped from $2.4\times10^{-2}$ to $6.6\times10^{-3}$, and the optimal mixing is $\alpha=0.3885$. The 16 entries of `zvo_ls_qqqq_001.dat` begin
`1, -8.6042, -8.6042, 75.79, -8.6042, 75.76, 75.79, ...`; entries 2, 3, 10, 11, 15 (0-based) are $h_1,h_{2(11)},h_{2(20)},h_{3(12)},h_4$
([6.2](06-theory-observables-lanczos.md#62-the-single-step-lanczos-wave-function)). As it must, $h_{2(11)}=75.79059\ldots$ equals the third column of `zvo_out_001.dat`
($\langle\lvert E_{\rm loc}\rvert^2\rangle$), and entry 10, $h_{2(20)}=75.76$, is the separately measured $\langle F^\dagger(H^2)\rangle$.

## 10.5 Variations

- **CG solver.** Append `NSRCG 1` to `modpara.def` and run 5 steps (`--nsteps 5 --nsmp 5`): a `zvo_SRinfo.dat` appears with, per step, `Npara Msize optCut diagCut sDiagMax sDiagMin absRmax imax, iterations`:

  ```text
  #Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax
    100   100     0     0  1.22546e+00  1.14682e-02 -6.31767e-02    76, 100
    100   100     0     0  1.40307e+00  1.05528e-02 -6.39915e-02    76, 100
  ```

  Here the CG solver ran its maximum of 100 iterations (`NSROptCGMaxIter` default $=n_S$), i.e. it did not reach `DSROptCGTol`.
- **Timers.** `MVMC_C_TIMER=1 mvmc hubbard_L16/namelist.def --nsteps 30 --nsmp 5 --out-dir out_t` additionally writes `zvo_CalcTimer.dat`
  ([9.5](09-output-files.md#95-timers)); for this model most of the time is in `hopping update` and `UpdateMAll`.
- **Rejected input.** Appending `NSRCG 2` makes `mvmc` stop with `error: NSRCG >= 2 is not supported by Julia-mVMC; use NSRCG = 0 or 1` and exit status 1.
- **Reproducibility.** Running the same command twice with `--seed 5` gives byte-identical `zvo_out.dat`.

The example programs (`cargo run -p mvmc-cli --example hubbard_chain`, `heisenberg_chain_real`, `heisenberg_chain_cmp`, `heisenberg_chain_fsz`) run the inputs of the Julia reference
(`extern/Julia-mVMC/examples/inputs`, found through the submodule or `JULIA_MVMC_ROOT`) for a few SR steps (`JULIA_MVMC_EXAMPLE_STEPS`, `MVMC_OUT_DIR`); they were not run for this manual **(unverified)**.
