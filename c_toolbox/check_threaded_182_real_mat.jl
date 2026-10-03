# Explicit developer-only, fixed-operand C-real FSZ matrix stage probe.
# No Cargo dependency, fixture publication, sampling or RNG claims.
using SHA, Libdl, LinearAlgebra, PfaPack, Printf
using MVMCOptimizers, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 1 || error("usage: NEW-external-artifact-directory")
repo = realpath(dirname(@__DIR__))
requested = abspath(ARGS[1])
dest = joinpath(realpath(dirname(requested)), basename(requested))
startswith(dest * "/", repo * "/") && error("external artifacts required")
(ispath(dest) || islink(dest)) && error("fresh directory required")
source_root = joinpath(repo, "extern/mVMC-1.3.0/src")
matrix_path = joinpath(source_root, "mVMC/matrix.c")
matrix = read(matrix_path, String)
bytes2hex(sha256(matrix)) == "849488176375102fbc3bab7001e0dc27dfa7c8e049fd90ee1d6248ebfba10764" || error("C matrix source changed")
start = findfirst("int calculateMAll_child_fsz_real(", matrix)
start === nothing && error("C child missing")
stop = findnext("\n}", matrix, first(start))
stop === nothing && error("C child boundary missing")
body = matrix[first(start):last(stop)]
qp_path = joinpath(source_root, "mVMC/qp_real.c")
qp = read(qp_path, String)
bytes2hex(sha256(qp)) == "eec104362f5540c368aed7e42403a79eeefc21a509da6c57c668b5c4fe887b71" || error("C real IP source changed")
qp_start = findfirst("double  CalculateIP_real(", qp)
qp_start === nothing && error("C real IP missing")
qp_stop = findnext("\n}", qp, first(qp_start))
qp_stop === nothing && error("C real IP boundary missing")
ip_body = qp[first(qp_start):last(qp_stop)]
slater_path = joinpath(source_root,"mVMC/slater_fsz.c")
slater_source = read(slater_path,String)
bytes2hex(sha256(slater_source)) == "2bb20af97f47c30cf2c81a856ee7235df5f45316546c2ae98b43785d99b90eec" || error("C differential source changed")
diff_signature = "void SlaterElmDiff_fsz(double complex *srOptO, const double complex ip, int *eleIdx,int *eleSpn) {"
length(findall(diff_signature,slater_source)) == 1 || error("C differential definition boundary changed")
diff_start = findfirst(diff_signature,slater_source)
diff_start === nothing && error("C differential missing")
diff_stop = findnext("\n}",slater_source,first(diff_start))
diff_stop === nothing && error("C differential boundary missing")
diff_body = slater_source[first(diff_start):last(diff_stop)]
main_path = joinpath(source_root,"mVMC/vmccal_fsz.c")
main_source = read(main_path,String)
bytes2hex(sha256(main_source)) == "9ac529ef2a18d80f61c5aca92aacce10391e2794ecc56aa8ca947691dbdd974e" || error("C shadow/even-O source changed")
function source_line(text,needle)
    lines = filter(line->occursin(needle,line),split(text,'\n'))
    length(lines)==1 || error("C extraction boundary changed: $needle")
    only(lines)
end
shadow_copy = source_line(main_source,"for(tmp_i=0;tmp_i<NQPFull*(Nsize*Nsize+1);tmp_i++)")
even_copy = source_line(main_source,"srOptO_real[i] = creal(srOptO[2*i]);")
notice_end = findfirst("*/", matrix)
notice_end === nothing && error("C license missing")
notice = matrix[1:last(notice_end)]
mkdir(dest)
BLAS.set_num_threads(1)
ENV["OPENBLAS_NUM_THREADS"] = "1"
commands = String[]
inputs = String[matrix_path, qp_path,slater_path,main_path]
function execute(command)
    push!(commands, string(command))
    run(command)
end
objects = String[]
for name in ("dsktrf.f", "dsktf2.f", "dlasktrf.f", "dskr2.f", "dskr2k.f")
    path = joinpath(source_root, "pfapack/fortran", name)
    push!(inputs, path)
    object = joinpath(dest, name * ".o")
    execute(`gfortran -O0 -ffp-contract=off -fPIC -c $path -o $object`)
    push!(objects, object)
end
ltl = joinpath(source_root, "ltl2inv")
common = joinpath(source_root, "common")
for name in ("ltl2inv.cc", "ilaenv_lauum.cc")
    path = joinpath(ltl, name)
    object = joinpath(dest, name * ".o")
    execute(`c++ -std=c++11 -O0 -ffp-contract=off -fPIC -DBLAS_EXTERNAL -I$common -I$(joinpath(common, "deps")) -I$ltl -c $path -o $object`)
    push!(objects, object)
end
wrap = joinpath(ltl, "ilaenv_wrap.f90")
object = joinpath(dest, "ilaenv_wrap.o")
execute(`gfortran -O0 -ffp-contract=off -fPIC -c $wrap -o $object`)
push!(objects, object)
# Hash all local transitive headers/templates as well as compiled units.
for directory in (ltl, common)
    for (root, _, files) in walkdir(directory), name in files
        push!(inputs, joinpath(root, name))
    end
end
before = Dict(path => bytes2hex(sha256(read(path))) for path in unique(inputs))
driver = notice * """
/* Origin: mVMC-1.3.0/src/mVMC/matrix.c:223-278, GPL-3.0-or-later.
 * Extracted child verbatim; upstream utu2pfa/utu2inv and PfaPack linked.
 * Synthetic fixed operands only; not a full input/runner oracle. */
#include <math.h>
#include <stdlib.h>
#include <complex.h>
int Nsite, Nsite2, Nsize;
double *SlaterElm_real, *InvM_real, *PfM_real;
extern int dsktrf_(const char*, const char*, const int*, double*, const int*, int*, double*, const int*, int*);
extern void dscal_(const int*, const double*, double*, const int*);
extern void utu2pfa_d(int, double*, int, int*, double*);
extern void utu2inv_d(int, double*, int, int*, double*, double*, int);
#define M_DSKTRF dsktrf_
#define M_DSCAL dscal_
$body
/* Origin: qp_real.c:53, GPL-3.0-or-later, same upstream notice above.
 * MPI_COMM_SELF size-one stub only; any collective attempt aborts. */
typedef int MPI_Comm;
#define MPI_DOUBLE 0
#define MPI_SUM 0
double complex *QPFullWeight;
int MPI_Comm_size(MPI_Comm comm, int *size) { (void)comm; *size=1; return 0; }
int MPI_Allreduce(const void *in, void *out, int n, int type, int op, MPI_Comm comm) {
  (void)in;(void)out;(void)n;(void)type;(void)op;(void)comm; abort();
}
$ip_body
double probe_real_ip(int nq, double *pf, double complex *weights) {
  QPFullWeight=weights;
  return CalculateIP_real(pf,0,nq,0);
}
int NQPFull, NSlater, NMPTrans, NSPGaussLeg, NQPOptTrans;
int **OrbitalIdx, **OrbitalSgn, **QPTrans, **QPTransSgn;
int **QPOptTrans, **QPOptTransSgn;
double complex *InvM, *PfM;
static int *integer_scratch, integer_offset;
static double complex *complex_scratch;
void RequestWorkSpaceInt(int n) { integer_scratch=calloc(n,sizeof(int)); integer_offset=0; if(!integer_scratch) abort(); }
void RequestWorkSpaceComplex(int n) { complex_scratch=calloc(n,sizeof(double complex)); if(!complex_scratch) abort(); }
int *GetWorkSpaceInt(int n) { int *p=integer_scratch+integer_offset; integer_offset+=n; return p; }
double complex *GetWorkSpaceComplex(int n) { (void)n; return complex_scratch; }
void ReleaseWorkSpaceInt(void) { free(integer_scratch); }
void ReleaseWorkSpaceComplex(void) { free(complex_scratch); }
/* Origin: slater_fsz.c:119-245 and vmccal_fsz.c shadow/even-O boundaries,
 * GPL-3.0-or-later. Scratch allocation plumbing is harness-only. */
$diff_body
void probe_real_diff(int ns, int size, int nq, double *inverse_and_pf,
                     int *idx, int *spins, double complex *weights,
                     double ip, double complex *output, double *even_output) {
  Nsite=ns; Nsite2=2*ns; Nsize=size;
  NQPFull=nq; NMPTrans=nq; NSPGaussLeg=1; NQPOptTrans=1;
  NSlater=Nsite2*Nsite2; QPFullWeight=weights;
  InvM_real=inverse_and_pf;
  InvM=malloc(sizeof(double complex)*NQPFull*(Nsize*Nsize+1));
  if(!InvM) abort(); PfM=InvM+NQPFull*Nsize*Nsize;
  int tmp_i;
$shadow_copy
  OrbitalIdx=malloc(Nsite2*sizeof(int*)); OrbitalSgn=malloc(Nsite2*sizeof(int*));
  for(int r=0;r<Nsite2;++r) {
    OrbitalIdx[r]=malloc(Nsite2*sizeof(int)); OrbitalSgn[r]=malloc(Nsite2*sizeof(int));
    for(int c=0;c<Nsite2;++c) { OrbitalIdx[r][c]=r*Nsite2+c; OrbitalSgn[r][c]=1; }
  }
  QPTrans=malloc(nq*sizeof(int*)); QPTransSgn=malloc(nq*sizeof(int*));
  for(int q=0;q<nq;++q) {
    QPTrans[q]=malloc(ns*sizeof(int)); QPTransSgn[q]=malloc(ns*sizeof(int));
    for(int r=0;r<ns;++r) { QPTrans[q][r]=r; QPTransSgn[q][r]=1; }
  }
  QPOptTrans=QPTrans; QPOptTransSgn=QPTransSgn;
  int SROptSize=3+NSlater;
  double complex *srOptO=output;
  double *srOptO_real=even_output;
  for(int i=0;i<2*SROptSize;++i) srOptO[i]=0;
  srOptO[0]=1; srOptO[2]=3; srOptO[4]=-2;
  SlaterElmDiff_fsz(srOptO+6,ip,idx,spins);
  for(int i=0;i<SROptSize;++i) {
$even_copy
  }
  free(InvM);
  for(int r=0;r<Nsite2;++r) { free(OrbitalIdx[r]); free(OrbitalSgn[r]); }
  free(OrbitalIdx); free(OrbitalSgn);
  for(int q=0;q<nq;++q) { free(QPTrans[q]); free(QPTransSgn[q]); }
  free(QPTrans); free(QPTransSgn);
}
int probe_real_mat(int ns, int size, int nq, double *slater,
                   int *idx, int *spins, double *inverse_and_pf) {
  Nsite=ns; Nsite2=2*ns; Nsize=size;
  SlaterElm_real=slater; InvM_real=inverse_and_pf;
  PfM_real=InvM_real+nq*size*size;
  double *buf=calloc(size*size,sizeof(double));
  double *work=calloc(size*size,sizeof(double));
  int *piv=calloc(size,sizeof(int));
  if(!buf || !work || !piv) abort();
  int info=0;
  for(int q=0;q<nq;++q) {
    info=calculateMAll_child_fsz_real(idx,spins,0,nq,q,buf,piv,work,size*size);
    if(info) break;
  }
  free(buf); free(work); free(piv);
  return info;
}
"""
driver_path = joinpath(dest, "probe.c")
write(driver_path, driver)
object = joinpath(dest, "probe.o")
execute(`cc -std=c11 -O0 -ffp-contract=off -fPIC -Wno-unknown-pragmas -c $driver_path -o $object`)
push!(objects, object)
library = joinpath(dest, "libreal_mat." * Libdl.dlext)
execute(`c++ -shared $objects -lopenblas -lgfortran -o $library`)
handle = Libdl.dlopen(library)
call = Libdl.dlsym(handle, :probe_real_mat)
ip_call = Libdl.dlsym(handle, :probe_real_ip)
diff_call = Libdl.dlsym(handle,:probe_real_diff)
open(joinpath(dest, "stage.log"), "w") do io
    for size in (2, 4, 6, 8)
        ns, nq = size, 3
        idx = Cint.(0:size-1)
        spins = Cint.(mod.(0:size-1, 2))
        slater = zeros(Float64, nq*(2ns)^2)
        for q in 0:nq-1, i in 0:2ns-1, j in i+1:2ns-1
            value = ((17i+13j+7q)%31-15)/7 + 0.125
            slater[q*(2ns)^2+i*2ns+j+1] = value
            slater[q*(2ns)^2+j*2ns+i+1] = -value
        end
        native = fill(NaN, nq*(size^2+1))
        operands_before = (copy(slater), copy(idx), copy(spins))
        info = ccall(call, Cint, (Cint,Cint,Cint,Ptr{Float64},Ptr{Cint},Ptr{Cint},Ptr{Float64}), ns,size,nq,slater,idx,spins,native)
        info == 0 || error("C real factorization failure: size=$size info=$info")
        (slater,idx,spins) == operands_before || error("C matrix mutated borrowed operands")
        pf_values = native[nq*size^2+1:end]
        weights = ComplexF64[1.0,0.5,-0.25]
        data = MVMCExpertModeParsers.ExpertModeData()
        data.qp_weights = MVMCExpertModeParsers.QuantumProjectionWeights()
        data.qp_weights.qp_full_weight = weights
        c_ip = ccall(ip_call, Float64, (Cint,Ptr{Float64},Ptr{ComplexF64}),nq,pf_values,weights)
        j_ip = MVMCOptimizers.calculate_ip_real(pf_values,1,nq+1,data)
        abs(c_ip-j_ip) <= 1e-13+1e-13*max(abs(c_ip),abs(j_ip)) || error("first divergence real IP size=$size")
        @printf(io,"size=%d real_ip_error=%.18e C_ip=%.18e Julia_ip=%.18e\n",size,abs(c_ip-j_ip),c_ip,j_ip)
        nslater = (2ns)^2
        data.modpara.nsite=ns; data.modpara.nelec=size÷2
        data.modpara.n_orbital_idx=nslater
        data.modpara.nmp_trans=nq; data.modpara.nsp_gauss_leg=1
        data.n_qp_opt_trans=1
        data.qp_trans=[collect(0:ns-1) for _ in 1:nq]
        data.qp_trans_sgn=[ones(Int,ns) for _ in 1:nq]
        data.qp_opt_trans=[collect(0:ns-1)]
        data.qp_opt_trans_sgn=[ones(Int,ns)]
        data.orbital_idx_matrix=[(r-1)*2ns+c-1 for r in 1:2ns,c in 1:2ns]
        data.orbital_sgn=ones(Int,2ns,2ns)
        state=MVMCOptimizers.VMCOptimizationState(ns,size÷2,2,nslater,nq,1,false,true)
        state.slater_matrix.inv_m[1:nq*size^2] .= ComplexF64.(native[1:nq*size^2])
        state.slater_matrix.pf_m .= ComplexF64.(pf_values)
        state.slater_matrix.inv_m_real[1:nq*size^2] .= native[1:nq*size^2]
        state.slater_matrix.pf_m_real .= pf_values
        state.slater_matrix.inv_m[1:nq*size^2] == ComplexF64.(native[1:nq*size^2]) || error("inverse shadow copy failed")
        state.slater_matrix.pf_m == ComplexF64.(pf_values) || error("PfM shadow copy failed")
        c_o=zeros(ComplexF64,2*(3+nslater)); c_even=zeros(3+nslater)
        native_before=copy(native)
        ccall(diff_call,Cvoid,(Cint,Cint,Cint,Ptr{Float64},Ptr{Cint},Ptr{Cint},Ptr{ComplexF64},Float64,Ptr{ComplexF64},Ptr{Float64}),ns,size,nq,native,idx,spins,weights,c_ip,c_o,c_even)
        native == native_before || error("C differential mutated real matrix operands")
        j_o=zeros(ComplexF64,2*(3+nslater)); j_o[1]=1; j_o[3]=3; j_o[5]=-2
        MVMCOptimizers.slater_elm_diff_fsz!(@view(j_o[7:end]),ComplexF64(j_ip),Int.(idx),Int.(spins),data,state)
        all(abs.(j_o-c_o) .<= 1e-13 .+ 1e-13 .* max.(abs.(j_o),abs.(c_o))) || error("first divergence original FSZ differential size=$size")
        j_even=real.(j_o[1:2:end])
        all(abs.(j_even-c_even) .<= 1e-13 .+ 1e-13 .* max.(abs.(j_even),abs.(c_even))) || error("first divergence even-O size=$size")
        @printf(io,"size=%d differential_error=%.18e even_O_error=%.18e full_slots=%d\n",size,maximum(abs.(j_o-c_o)),maximum(abs.(j_even-c_even)),length(j_o))
        flush(io)
        for q in 0:nq-1
            a = [-slater[q*(2ns)^2+(Int(idx[i])+Int(spins[i])*ns)*2ns+Int(idx[j])+Int(spins[j])*ns+1] for j in 1:size, i in 1:size]
            original = copy(a)
            piv = zeros(Int, size)
            PfaPack.julia_dsktf2!(a, piv) == 0 || error("Julia real factorization failed")
            pf = PfaPack.utu2pfa(size, a, size, piv)
            PfaPack.utu2inv!(size, a, size, piv, zeros(size-1), zeros(size,size), size)
            a .*= -1.0
            c_inverse = reshape(native[q*size^2+1:(q+1)*size^2],size,size)
            c_pf = native[nq*size^2+q+1]
            inverse_error = maximum(abs.(a-c_inverse))
            pf_error = abs(pf-c_pf)
            # Existing fixed matrix policy: absolute+relative 1e-13.
            all(abs.(a-c_inverse) .<= 1e-13 .+ 1e-13 .* max.(abs.(a),abs.(c_inverse))) || error("first divergence C/Julia real inverse size=$size QP=$q")
            pf_error <= 1e-13 + 1e-13*max(abs(pf),abs(c_pf)) || error("first divergence real Pfaffian size=$size QP=$q")
            residual = opnorm(original*c_inverse + I, Inf)
            @printf(io,"size=%d QP=%d pf_error=%.18e inverse_error=%.18e inverse_residual=%.18e condition=%.18e\n",size,q,pf_error,inverse_error,residual,cond(original))
            flush(io)
        end
    end
end
for (path, hash) in before
    bytes2hex(sha256(read(path))) == hash || error("source changed during probe: $path")
end
open(joinpath(dest,"provenance.txt"),"w") do io
    println(io,"scope=12-fixed-operand-real-C-child-vs-Julia-real-PfaPack-stages plus 4 real-IP/differential/even-O stages; not-public-model, sampling, RNG or OO/HO")
    println(io,"Julia=$VERSION platform=$(Sys.MACHINE) BLAS=$(BLAS.get_config()) BLAS_threads=$(BLAS.get_num_threads())")
    println(io,"C extracted matrix.c:223-278; actual upstream real DSKTRF/utu2pfa_d/utu2inv_d/DSCAL; source-dependent stages not rewritten")
    println(io,"IP=source-guarded original CalculateIP_real with size-one MPI stub; collectives abort; NOT MPI protocol evidence")
    println(io,"policy=existing fixed-matrix absolute+relative 1e-13; residual/conditioning recorded, no tolerance increase")
    for command in commands; println(io,"command=",command); end
    for path in sort(collect(keys(before))); println(io,relpath(path,repo)," sha256=",before[path]); end
    for path in (driver_path,library,@__FILE__); println(io,basename(path)," sha256=",bytes2hex(sha256(read(path)))); end
end
println("PASS: 12 real C/Julia fixed-matrix plus 4 real-IP/differential/even-O stages; artifacts=$dest")
