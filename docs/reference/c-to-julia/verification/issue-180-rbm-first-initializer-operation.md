# First observed canonical complex initializer divergence

Independent optional probes `c_toolbox/reviewed_rbm_initializer_arithmetic.c`
and `.jl` locate the first arithmetic difference for declared slot6 (first
active RBM coefficient) at seed12395. Neither modifies an initializer,
transcendental, sampler or solver. No Rust output supplies expected values.
Normal Cargo tests never execute these probes.

C uses original `parameter.c:53`
`1e-2*genrand_real2()*cexp(2.0*I*M_PI*genrand_real2())`.
Julia's initializer draws r1 then r2 and evaluates
`ComplexF64(1e-2*r1*exp(2.0im*pi*r2))`. The standalone Julia probe uses that
unchanged expression; it is not a new fork acquisition or full draw-count
proof. The C probe includes the original unchanged InitParameter body and
also prints a decomposed radius/phase/unit calculation. Its original call
consumes 192 primitive words and reproduces the C acquisition's slot6.

## Actual results

Julia session36147 terminal0, chunkcc9d23. C invocation terminal0,
chunk5d27a6 (initial compile/run chunkc84694 also terminal0).
Full actual stdout is preserved in `evidence/issue-180-rbm-initializer-c.stdout.txt`
(SHA256 `69ea50dab4bef9c7d101b7d590763d7e6619e33078ca9cc70e7fa7d5a56b7dac`)
and `evidence/issue-180-rbm-initializer-julia-36147.stdout.txt`
(SHA256 `29da508b5f81c75e91289539662c4cb65f49cc1dddbd9acb01b06ce01e884724`).
Hexadecimal bits here identify diagnostic operations, not a bitwise
Monte Carlo numerical acceptance rule.

| Boundary | C bits | Julia bits |
| --- | --- | --- |
| First SFMT real2 conversion r1 | 3fb352ca9d000000 | same |
| Second SFMT real2 conversion r2 | 3fec29bdb9c00000 | same |
| Radius 1e-2*r1 | 3f48bbe4a0000000 | same |
| Phase imaginary part 2*pi*r2 | 40161e8476574965 | same |
| Unit complex exponential real part | 3fe75711d1dc4b51 | same |
| Unit complex exponential imaginary part | bfe5e43b2eef3f2d | bfe5e43b2eef3f2e |
| Radius-scaled imaginary coefficient | bf40ebb684a88bce | bf40ebb684a88bcf |

The **first observed differing operation is the imaginary component of the
unit complex exponential**, one ULP at this argument. It is not SFMT
conversion, radius/phase scaling, or a reordered multiplication. Radius is
7.548204739578068256e-4, phase5.529802178456837858. C's unit imaginary part
is -6.841102520647602825e-1; Julia's is -6.841102520647603935e-1.
Scaling produces C -5.163804247029170260e-4 versus Julia
-5.163804247029171344e-4, difference1.0842021724855044e-19.

These final numbers reproduce the actual checked-in C initialized slot6 and
the owner's reviewed-fork v2 initialized slot6 (stage
`/tmp/mvmc-review62b-v2-general-rbm-prefix123/parameter-audit.tsv`, SHA256
`36876780ba0aafd4cf28a093820f0ee28beabd753b1e19a2efab93bb2e5b505e`).
That stage's Bool flag serialization is separately classified as encoding;
it does not alter this arithmetic observation. The stage declares reviewed
fork62b0f97f076fb55c71c3ab0caa041a9adff94e04.

Julia1.13.1 `Base.exp(z::Complex)` (`complex.jl:694–713`) computes exp(real),
then sincos(imag), then the two scalar products. `Base.Math.sincos` for
Float64 (`special/trig.jl:177–203`) uses `rem_pio2_kernel` and
`sincos_kernel` plus quadrant selection; it is not simply a call to glibc
sincos. The probe's direct Julia sincos sine equals its complex-exp imaginary
component. C calls the container's libm cexp. Internal libm-versus-Julia
range-reduction/kernel operations were **not** individually instrumented;
the localization is the initializer's exponential boundary, not a fabricated
claim about the first internal trig instruction.

## Source/environment/reproduction

Linux x86_64 container73c57e563c61; GCC13.3.0; Julia1.13.1;
BLAS none in this arithmetic probe. Original C source/excerpt/license and
compiler dependencies are documented in
[the C parameter acquisition](../../../../c_toolbox/reviewed_parameter_c_audit.md).
C diagnostic adapter SHA256
`ce931d78ddf6a6408a17596535d17b1a5fe44f40ee0d80032d5a9e11bae9a392`;
Julia diagnostic SHA256
`6cda57b3530b282b3791bcabef3549927cfd4906e3a3ee8e406d35e90c07da7a`.
C probe executable SHA256
`51d08727606606522d4eb31e759ef9e2d0bbda3d90657e05ad119ba76f4f305a`;
linked `/lib/x86_64-linux-gnu/libm.so.6` SHA256
`f06f2ce1f1833df5f41cf13b6447ff07bea993ad9b27297d3428c2f70ab3f0e7`.
Installed Julia complex.jl SHA256
`bcff28dcbb4b88e2340bf01d3696a935738c4b857cfef6813bf333a7efcb4579`;
special/trig.jl SHA256
`194a4091a297431b017b9e80e7f2e3eb4cd88be75647a7d5409f873523594a7d`.

```sh
gcc -std=gnu11 -O0 -ffp-contract=off -DMEXP=19937 -I/workspaces/mvmc-rs/extern/mVMC-1.3.0/src/sfmt -I/workspaces/mvmc-rs/extern/mVMC-1.3.0/src/mVMC/include -I/home/vscode/.cache/mvmc/reviewed-c-parameter.AFmMKn/canonical-masked-groups3 -I/workspaces/mvmc-rs/c_toolbox /workspaces/mvmc-rs/c_toolbox/reviewed_rbm_initializer_arithmetic.c -lm -o /home/vscode/.cache/mvmc/reviewed-c-parameter.AFmMKn/rbm-arithmetic
/home/vscode/.cache/mvmc/reviewed-c-parameter.AFmMKn/rbm-arithmetic
/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia --compiled-modules=existing --project=/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8/extern/Julia-mVMC /workspaces/mvmc-rs/c_toolbox/reviewed_rbm_initializer_arithmetic.jl
```

At the time of the standalone probe, no tolerance had been assigned and the
full-state comparison was pending. Subsequently, the fresh reviewed-fork
stage `/tmp/mvmc-review62b-state.nvDReR-stage` completed acquisition52531
(terminal0). Owner audit69494 and independent parent verification23898
proved exact raw624-word state, index192 and primitive count192 at all nine
phase boundaries, exact mapping for918 records, and1782 defined C flag
comparisons. Six undefined C flag cells per phase remain masked. The earlier
v2 stage above is retained as the lineage of the first arithmetic observation.

The user approved absolute bound8.7e-19 **only for this initialized/overlaid/
synchronized parameter audit**. This is not a global numerical rule, an SR
solution bound, or a CG tolerance. The standalone first-two conversion probe
is not the evidence for the later full-state result. No unwritten-zero parity
or full-C sampler claim follows.
