# Optional C numerical kernel check. No Rust-derived input or expectation.
using SHA, Printf
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
repo = dirname(@__DIR__)
root = joinpath(repo, "tests/fixtures/physcal_181")
source_path = joinpath(repo, "extern/mVMC-1.3.0/src/mVMC/average.c")
source = read(source_path, String)
first_pos = first(findlast("void weightAverageReduce_fcmp(int n", source))
last_pos = first(findnext("void weightAverageReduce_real(int n", source, first_pos)) - 1
kernel = source[first_pos:last_pos]
driver = """
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
typedef int MPI_Comm;
static double complex Wc;
static void MPI_Comm_rank(MPI_Comm c,int *r) {(void)c;*r=0;}
static void MPI_Comm_size(MPI_Comm c,int *s) {(void)c;*s=1;}
static void RequestWorkSpaceComplex(int n) {(void)n;abort();}
static double complex *GetWorkSpaceComplex(int n) {(void)n;abort();}
static void SafeMpiReduce_fcmp(double complex *a,double complex *b,int n,MPI_Comm c) {(void)a;(void)b;(void)n;(void)c;abort();}
static void ReleaseWorkSpaceComplex(void) {abort();}
""" * kernel * """
int main(void) {
  double re,im; int n;
  if(scanf("%lf %lf %d", &re,&im,&n)!=3 || n<0) return 2;
  Wc=CMPLX(re,im);
  double complex *values=calloc((size_t)n,sizeof(*values));
  if(n && !values) return 3;
  for(int i=0;i<n;i++) {
    if(scanf("%lf %lf",&re,&im)!=2) return 4;
    values[i]=CMPLX(re,im);
  }
  weightAverageReduce_fcmp(n,values,0);
  for(int i=0;i<n;i++) printf("%.17g %.17g\\n",creal(values[i]),cimag(values[i]));
  free(values); return 0;
}
"""
function pairs(path)
    values = parse.(Float64, split(read(path, String)))
    iseven(length(values)) || error("incomplete complex record: $path")
    ComplexF64[complex(values[i], values[i+1]) for i in 1:2:length(values)]
end
destination = joinpath(root, "native-c-weighted-green")
mkpath(destination)
mktempdir() do work
    path = joinpath(work, "probe.c"); write(path, driver)
    binary = joinpath(work, "probe")
    run(`cc -std=c11 -O0 -ffp-contract=off $path -o $binary`)
    report = String[]
    for model in sort(readdir(joinpath(root, "two-samples"))), frame in 0:1
        input = joinpath(root, "two-samples", model, "accumulated-$frame")
        averaged = joinpath(root, "two-samples", model, "averaged-$frame")
        weight = first(pairs(joinpath(input, "energy.txt")))
        for family in ("one", "factored", "direct")
            sums = pairs(joinpath(input, "$family.txt"))
            request = "$(real(weight)) $(imag(weight)) $(length(sums))\n" *
                join(["$(real(v)) $(imag(v))\n" for v in sums])
            result = read(pipeline(`$binary`, stdin=IOBuffer(request)), String)
            output = joinpath(destination, model, "frame-$frame")
            mkpath(output); write(joinpath(output, "$family.txt"), result)
            native = pairs(joinpath(output, "$family.txt"))
            julia = pairs(joinpath(averaged, "$family.txt"))
            length(native) == length(julia) || error("shape divergence")
            first_difference = "none"
            max_abs = 0.0
            for (index, (c, j)) in enumerate(zip(native, julia)), component in (:re, :im)
                cvalue, jvalue = getproperty(c, component), getproperty(j, component)
                delta = cvalue-jvalue
                max_abs = max(max_abs, abs(delta))
                if delta != 0.0 && first_difference == "none"
                    first_difference = "index=$(index-1) component=$component C=$(repr(cvalue)) Julia=$(repr(jvalue)) delta=$(repr(delta))"
                end
                abs(delta) <= max(1e-12, 1e-10*max(abs(cvalue),abs(jvalue))) ||
                    error("C-J normalization divergence: $model frame$frame $family $first_difference")
            end
            push!(report, "$model frame=$frame family=$family max_abs=$max_abs first_difference=$first_difference")
            push!(report, "input_sha256=$(bytes2hex(sha256(read(joinpath(input, "$family.txt"))))) $(relpath(joinpath(input, "$family.txt"),root))")
            push!(report, "weight_input_sha256=$(bytes2hex(sha256(read(joinpath(input, "energy.txt"))))) $(relpath(joinpath(input, "energy.txt"),root))")
        end
    end
    # Also verify the deliberately non-real weight in the independent
    # standalone fixture; actual serial PhysCal frames have unit sample weights.
    synthetic_input = pairs(joinpath(root, "weighted-average/accumulated.txt"))
    synthetic_expected = pairs(joinpath(root, "weighted-average/averaged.txt"))[6:end]
    weight = synthetic_input[1]
    reciprocal = read(pipeline(`$binary`, stdin=IOBuffer(
        "$(real(weight)) $(imag(weight)) 1\n1.0 0.0\n")), String)
    inverse_parts = parse.(Float64, split(reciprocal))
    inverse_c = complex(inverse_parts...)
    inverse_julia = 1.0 / weight
    write(joinpath(destination, "reciprocal-first-divergence.txt"),
        "weight=$(repr(weight))\nC_reciprocal=$(repr(inverse_c))\nJulia_reciprocal=$(repr(inverse_julia))\n" *
        "delta_re=$(repr(real(inverse_c)-real(inverse_julia)))\ndelta_im=$(repr(imag(inverse_c)-imag(inverse_julia)))\n" *
        "scope=actual C weightAverageReduce_fcmp on unit numerator, before any nontrivial Green multiplication; independent Julia reciprocal\n")
    sums = synthetic_input[6:end]
    request = "$(real(weight)) $(imag(weight)) $(length(sums))\n" *
        join(["$(real(v)) $(imag(v))\n" for v in sums])
    result = read(pipeline(`$binary`, stdin=IOBuffer(request)), String)
    write(joinpath(destination, "synthetic-green.txt"), result)
    synthetic_native = pairs(joinpath(destination, "synthetic-green.txt"))
    first_difference = "none"
    max_abs = 0.0
    for (index, (c, j)) in enumerate(zip(synthetic_native, synthetic_expected)), component in (:re, :im)
        cvalue, jvalue = getproperty(c, component), getproperty(j, component)
        delta = cvalue-jvalue
        max_abs = max(max_abs, abs(delta))
        if delta != 0.0 && first_difference == "none"
            first_difference = "index=$(index-1) component=$component C=$(repr(cvalue)) Julia=$(repr(jvalue)) delta=$(repr(delta))"
        end
        abs(delta) <= max(1e-12, 1e-10*max(abs(cvalue),abs(jvalue))) || error("synthetic C-J normalization divergence")
    end
    push!(report, "synthetic complex_weight=$(repr(weight)) max_abs=$max_abs first_difference=$first_difference")
    push!(report, "input_sha256=$(bytes2hex(sha256(read(joinpath(root,"weighted-average/accumulated.txt"))))) weighted-average/accumulated.txt")
    write(joinpath(destination, "comparison.txt"), join(report, '\n') * "\n")
    write(joinpath(destination, "provenance.txt"),
        "source=extern/mVMC-1.3.0/src/mVMC/average.c\nsource_sha256=$(bytes2hex(sha256(read(source_path))))\n" *
        "generator_sha256=$(bytes2hex(sha256(read(@__FILE__))))\ndriver_sha256=$(bytes2hex(sha256(driver)))\n" *
        "boundaries=weightAverageReduce_fcmp through before weightAverageReduce_real\n" *
        "architecture=$(Sys.MACHINE)\njulia=$VERSION\nmanifest_sha256=$(bytes2hex(sha256(read(joinpath(repo,"extern/Julia-mVMC/Manifest-v1.13.toml")))))\n" *
        "compiler=$(strip(read(`cc --version`,String)))\noptions=-std=c11 -O0 -ffp-contract=off\n" *
        "scope=actual C serial normalization of independent Julia raw sums; MPI rank0 size1 adapters; no C sampling/measurement or BLAS; raw input provenance in sibling two-samples models\n" *
        "contract=one native complex reciprocal Wc, then ordered multiplication; existing 1e-12 absolute/1e-10 relative policy, no tolerance increase\n")
end
