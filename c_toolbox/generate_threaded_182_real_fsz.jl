# Explicit developer-only adapted real-FSZ oracle. Never invoked by Cargo.
# First scope: one public-derived real input, direct SR, NStore=0, prefix=1.
using MVMCOptimizers, MVMCExpertModeParsers, PfaPack, SFMT, Random
using SHA, Libdl, LinearAlgebra, Printf
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
Threads.nthreads()==1 || error("native reference globals require Julia threads=1")
length(ARGS)==3 || error("usage: NEW-external-destination verified-Mat-library-dir verified-real-energy-dir")
const REPO=realpath(dirname(@__DIR__))
const SOURCE=joinpath(REPO,"extern/Julia-mVMC/MVMCOptimizers.jl/src")
const DEST=joinpath(realpath(dirname(abspath(ARGS[1]))),basename(abspath(ARGS[1])))
startswith(DEST*"/",REPO*"/") && error("stage outside repository")
(ispath(DEST)||islink(DEST)) && error("fresh staging directory required")
const MATLIB=joinpath(abspath(ARGS[2]),"libreal_mat."*Libdl.dlext)
const ENERGYLIB=joinpath(abspath(ARGS[3]),"libmvmc_fsz_reference_real."*Libdl.dlext)
bytes2hex(sha256(read(MATLIB)))=="0217b63344df4e0683b9fdd96546c9a6bd571d4feb25fe799ec9b89586dc0478" || error("unverified Mat-stage library")
bytes2hex(sha256(read(ENERGYLIB)))=="a1906c51587d0b788e1c1763cb1e31a072d946870efb0cc7bcf6be4044a6e131" || error("unverified real-energy library")
const GUARDS=Dict(
    "vmc_sampling.jl"=>"d9d47beccbe947721ee45a41c04f5cd59937ac0ff62cc6d761ac0397863e2c38",
    "vmc_main_cal.jl"=>"16def1b75c5a8b36462884983c18a4422882b62d3bb45543baf699756b8839d0",
    "vmc_para_opt.jl"=>"a17e7cab2e3ce7e1e1046d621aead2743acc0a84a080df772c9a396b6260a316",
    "run_para_opt_from_namelist.jl"=>"138b43064e2f6b04077fa85652ca10d2e32cfa75342dd84c7e30ac56481b15bb",
)
for (name,hash) in GUARDS
    bytes2hex(sha256(read(joinpath(SOURCE,name))))==hash || error("Julia source changed: $name")
end
Base.active_project()==joinpath(REPO,"extern/Julia-mVMC/Project.toml") || error("wrong reference project")
BLAS.set_num_threads(1)
ENV["OPENBLAS_NUM_THREADS"]="1"
mkdir(DEST)
mkdir(joinpath(DEST,"clones"))
const LOG=open(joinpath(DEST,"stages.log"),"w")
const draws=Ref(0)
const mat_calls=Ref(0)
const measured_samples=Ref(0)
const PRE_SR=Ref{Any}(nothing)
const SNAPSHOT=Ref{Any}(nothing)
const C_OO=Ref(Float64[])
const C_HO=Ref(Float64[])
const C_WC=Ref(0.0)
const MATCALL=Libdl.dlsym(Libdl.dlopen(MATLIB),:probe_real_mat)
const IPCALL=Libdl.dlsym(Libdl.dlopen(MATLIB),:probe_real_ip)
function function_body(file,signature)
    source=read(joinpath(SOURCE,file),String)
    length(findall(signature,source))==1 || error("function boundary changed: $signature")
    a=first(findfirst(signature,source)); b=findnext("\nend\n",source,a)
    b===nothing && error("function end missing: $signature")
    source[a:last(b)]
end
function replace_once(body,needle,replacement)
    length(findall(needle,body))==1 || error("clone boundary changed: $needle")
    replace(body,needle=>replacement;count=1)
end
function install_clone(name,body)
    path=joinpath(DEST,"clones",name*".jl")
    write(path,"# Source-guarded C-compatible adaptation, NOT unmodified Julia parity.\n"*body)
    Base.include(MVMCOptimizers,path)
end
function integers(path,values); write(path,join(values," ")*"\n"); end
function numeric(path,values)
    write(path,join(repr.(collect(reinterpret(Float64,ComplexF64.(vec(values)))))," ")*"\n")
end
function checkpoint(stage,data,state=nothing)
    dir=joinpath(DEST,stage); mkdir(dir)
    integers(joinpath(dir,"draw-count.txt"),[draws[]])
    integers(joinpath(dir,"next624.txt"),SFMT.sfmt_dump_rand32(624))
    numeric(joinpath(dir,"parameters.txt"),MVMCOptimizers.pack_parameters(data))
    if stage != "seeded"
        integers(joinpath(dir,"julia-raw-flags.txt"),Int.(data.optimization_flags))
        integers(joinpath(dir,"c-written-mask.txt"),FLAG_MASK)
        integers(joinpath(dir,"defined-flags.txt"),DEFINED_FLAGS)
        length(data.optimization_flags)==length(DEFINED_FLAGS) || error("flag shape changed")
        # Only active real axes agree with this Julia API. Imaginary defaults
        # are retained raw, never presented as native C allocator contents.
        Int.(data.optimization_flags[1:2:end])==DEFINED_FLAGS[1:2:end] || error("active real flags changed")
    end
    if data.qp_weights!==nothing
        numeric(joinpath(dir,"qp_weights.txt"),data.qp_weights.qp_full_weight)
    end
    if state!==nothing
        cfg=state.electron_config
        for field in (:ele_idx,:ele_cfg,:ele_num,:ele_proj_cnt,:ele_spn,:burn_ele_idx)
            integers(joinpath(dir,string(field)*".txt"),getproperty(cfg,field))
        end
        integers(joinpath(dir,"counter.txt"),vcat(cfg.counter[1:9],cfg.counter[11]))
    end
end
@eval SFMT.C_API begin
    function gen_rand32(); Main.draws[]+=1; ccall((:gen_rand32,libsfmt),UInt32,()); end
    function genrand_real2(); Main.draws[]+=1; ccall((:genrand_real2,libsfmt),Cdouble,()); end
    gen_rand64()=error("unaccounted 64-bit draw")
    genrand_real1()=error("unaccounted real1 draw")
    genrand_real3()=error("unaccounted real3 draw")
    fill_array32(array,size)=error("unaccounted bulk draw")
    fill_array64(array,size)=error("unaccounted bulk draw")
end

# Fresh PUBLIC definitions, not a synthetic NMPTrans override. QPTrans's
# header dimensions the available maps; C readdef.c keeps the separately
# selected abs(ModPara NMPTrans) count. No initial.def/In overlay is invented.
donor=joinpath(REPO,"extern/Julia-mVMC/test/integration/reference/heisenberg_chain_fsz/inputs")
inputdir=joinpath(DEST,"inputs"); mkdir(inputdir)
input_hashes=Dict{String,String}()
for name in sort(readdir(donor))
    endswith(name,".def") || continue
    name=="initial.def" && continue
    original=read(joinpath(donor,name),String)
    input_hashes[name]=bytes2hex(sha256(original))
    text=replace(original,r"(?m)^(ComplexType\s+)1\s*$"=>s"\g<1>0")
    if name=="modpara.def"
        text=replace(text,r"(?m)^(NStore\s+)1\s*$"=>s"\g<1>0")
    end
    write(joinpath(inputdir,name),text)
end
input=joinpath(inputdir,"namelist.def")
parsed=MVMCExpertModeParsers.parse_expert_mode_files(input)
parsed.i_flg_orbital_general!=0 || error("public input is not FSZ")
!MVMCOptimizers.get_all_complex_flag(parsed) || error("public input did not select real mode")
parsed.modpara.nsrcg==0 && parsed.modpara.nstore_o==0 || error("first oracle scope is direct/no-store only")

# Actual C original differential + OO kernels; public tables are supplied by
# the independent Julia parser, not generated by Rust. Native globals serial.
croot=joinpath(REPO,"extern/mVMC-1.3.0/src/mVMC")
slater=read(joinpath(croot,"slater_fsz.c"),String)
cal=read(joinpath(croot,"vmccal.c"),String)
reader=read(joinpath(croot,"readdef.c"),String)
bytes2hex(sha256(reader))=="6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9" || error("C reader changed")
bytes2hex(sha256(slater))=="2bb20af97f47c30cf2c81a856ee7235df5f45316546c2ae98b43785d99b90eec" || error("C differential changed")
bytes2hex(sha256(cal))=="c2db5fd32c5c83be189ffd9fbb89684f0696bab7fff5f7d36abaa370274d9d2f" || error("C OO changed")
function c_function(source,signature)
    length(findall(signature,source))==1 || error("C definition boundary changed")
    a=first(findfirst(signature,source)); b=findnext("\n}",source,a)
    b===nothing && error("C definition end missing")
    source[a:last(b)]
end
diff=c_function(slater,"void SlaterElmDiff_fsz(double complex *srOptO, const double complex ip, int *eleIdx,int *eleSpn) {")
oo=c_function(cal,"void calculateOO_real(double *srOptOO, double *srOptHO, const double *srOptO,\n                 const double w, const double e, const int srOptSize) {")
flag_reader=c_function(reader,"int GetInfoOpt(FILE *fp, int *ArrayOpt, int iComplxFlag, int *iTotalOptCount, int fidx) {")
parallel_reader=c_function(reader,"int GetInfoOptOrbitalParalell(FILE *fp, int *ArrayOpt, int iComplxFlag, int *iTotalOptCount, int fidx) {")
notice=slater[1:last(findfirst("*/",slater))]
c_source=notice*"""
/* Extracted origin: mVMC-1.3.0/src/mVMC/slater_fsz.c and vmccal.c.
 * GPL-3.0-or-later; author Satoshi Morita. Serial harness plumbing only. */
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
$flag_reader
$parallel_reader
int public_flags(const char *rows,int *flags,int offset,int parallel){
 FILE *fp=tmpfile();if(!fp)abort();fputs(rows,fp);rewind(fp);int count=0;
 int result=parallel?GetInfoOptOrbitalParalell(fp,flags,0,&count,offset):GetInfoOpt(fp,flags,0,&count,offset);
 fclose(fp);return result;
}
int Nsite,Nsite2,Nsize,NSlater,NQPFull,NMPTrans,NSPGaussLeg,NQPOptTrans;
int **OrbitalIdx,**OrbitalSgn,**QPTrans,**QPTransSgn,**QPOptTrans,**QPOptTransSgn;
double complex *InvM,*PfM,*QPFullWeight;
static int *iw,ioff; static double complex *cw;
void RequestWorkSpaceInt(int n){iw=calloc(n,sizeof(int));ioff=0;if(!iw)abort();}
void RequestWorkSpaceComplex(int n){cw=calloc(n,sizeof(double complex));if(!cw)abort();}
int *GetWorkSpaceInt(int n){int*p=iw+ioff;ioff+=n;return p;}
double complex *GetWorkSpaceComplex(int n){(void)n;return cw;}
void ReleaseWorkSpaceInt(void){free(iw);}
void ReleaseWorkSpaceComplex(void){free(cw);}
extern void dger_(const int*,const int*,const double*,const double*,const int*,const double*,const int*,double*,const int*);
extern void daxpy_(const int*,const double*,const double*,const int*,double*,const int*);
#define M_DGER dger_
#define M_DAXPY daxpy_
$diff
$oo
void public_diff(int ns,int sz,int nq,int norb,int *orb,int *sign,
                 int *maps,int *mapsgn,int *idx,int *spins,
                 double complex *inverse,double complex *pf,double complex *weights,
                 double ip,double complex *out){
 Nsite=ns;Nsite2=2*ns;Nsize=sz;NSlater=norb;
 NQPFull=NMPTrans=nq;NSPGaussLeg=NQPOptTrans=1;
 OrbitalIdx=malloc(Nsite2*sizeof(int*));OrbitalSgn=malloc(Nsite2*sizeof(int*));
 for(int i=0;i<Nsite2;i++){OrbitalIdx[i]=orb+i*Nsite2;OrbitalSgn[i]=sign+i*Nsite2;}
 QPTrans=malloc(nq*sizeof(int*));QPTransSgn=malloc(nq*sizeof(int*));
 for(int q=0;q<nq;q++){QPTrans[q]=maps+q*ns;QPTransSgn[q]=mapsgn+q*ns;}
 int *identity=malloc(ns*sizeof(int)),*ones=malloc(ns*sizeof(int));
 for(int i=0;i<ns;i++){identity[i]=i;ones[i]=1;}
 QPOptTrans=&identity;QPOptTransSgn=&ones;
 InvM=inverse;PfM=pf;QPFullWeight=weights;
 SlaterElmDiff_fsz(out,ip,idx,spins);
 free(OrbitalIdx);free(OrbitalSgn);free(QPTrans);free(QPTransSgn);free(identity);free(ones);
}
"""
cpath=joinpath(DEST,"public_stages.c"); write(cpath,c_source)
clib=joinpath(DEST,"libpublic_stages."*Libdl.dlext)
run(`cc -std=c11 -O0 -ffp-contract=off -fPIC -shared -Wno-unknown-pragmas $cpath -lopenblas -o $clib`)
const STAGE_HANDLE=Libdl.dlopen(clib)
const DIFFCALL=Libdl.dlsym(STAGE_HANDLE,:public_diff)
const OOCALL=Libdl.dlsym(STAGE_HANDLE,:calculateOO_real)
const FLAGCALL=Libdl.dlsym(STAGE_HANDLE,:public_flags)
# Native reader writes are discovered using two distinct sentinels. Neither
# sentinel represents upstream malloc contents; unwritten slots remain unknown.
function native_flags(sentinel)
    flags=fill(Cint(sentinel),length(parsed.optimization_flags)); offset=0
    for (name,parallel) in [("gutzwilleridx.def",false),("jastrowidx.def",false),("orbitalidx.def",false),("orbitalidxpara.def",true)]
        lines=readlines(joinpath(inputdir,name))
        count=parse(Int,split(lines[2])[2])
        occursin(r"^ComplexType\s+0\s*$",lines[3]) || error("real flag header required")
        rows=join(lines[end-count+1:end],"\n")*"\n"
        got=ccall(FLAGCALL,Cint,(Cstring,Ptr{Cint},Cint,Cint),rows,flags,offset,parallel)
        got==count || error("C reader count mismatch")
        offset+=parallel ? 2count : count
    end
    2offset==length(flags) || error("public flag layout mismatch")
    flags
end
const FLAG_FIRST=native_flags(-701)
const FLAG_SECOND=native_flags(-709)
const FLAG_MASK=Int.(FLAG_FIRST.==FLAG_SECOND)
const DEFINED_FLAGS=[FLAG_MASK[i]==1 ? Int(FLAG_FIRST[i]) : 0 for i in eachindex(FLAG_MASK)]
all(FLAG_MASK[1:2:end].==1) || error("active real flag unwritten")
all(DEFINED_FLAGS[2:2:end].==0) || error("real inactive flag policy contradicted by native write")

function compare_stage(label,actual,expected;atol=1e-13,rtol=1e-13)
    length(actual)==length(expected) || error("shape mismatch $label")
    errors=abs.(actual.-expected)
    all(errors .<= atol .+ rtol.*max.(abs.(actual),abs.(expected))) || error("first divergence $label index=$(findfirst(errors .> atol .+ rtol.*max.(abs.(actual),abs.(expected))))")
    println(LOG,label," maximum_difference=",maximum(errors;init=0.0)); flush(LOG)
end
function real_mat!(idx,spins,qstart,qend,data,state)
    ns=data.modpara.nsite; size=length(idx); nq=length(state.slater_matrix.pf_m_real)
    qstart==1 && qend==nq+1 || error("first public oracle requires full QP range")
    slt=state.slater_matrix.slater_elm_real
    length(slt)==nq*(2ns)^2 || error("real Slater shape")
    native=zeros(nq*(size^2+1)); ci=Cint.(idx); cs=Cint.(spins)
    info=ccall(MATCALL,Cint,(Cint,Cint,Cint,Ptr{Float64},Ptr{Cint},Ptr{Cint},Ptr{Float64}),ns,size,nq,slt,ci,cs,native)
    info==0 || return Int(info)
    inverses=zeros(nq*size^2); pf=zeros(nq)
    for q in 0:nq-1
        a=[-slt[q*(2ns)^2+(idx[i]+spins[i]*ns)*2ns+idx[j]+spins[j]*ns+1] for j in 1:size,i in 1:size]
        piv=zeros(Int,size)
        PfaPack.julia_dsktf2!(a,piv)==0 || error("Julia real matrix failure")
        pf[q+1]=PfaPack.utu2pfa(size,a,size,piv)
        PfaPack.utu2inv!(size,a,size,piv,zeros(size-1),zeros(size,size),size)
        inverses[q*size^2+1:(q+1)*size^2].=-vec(a)
    end
    compare_stage("public_real_Mat_$(mat_calls[])",inverses,native[1:nq*size^2])
    compare_stage("public_real_PfM_$(mat_calls[])",pf,native[nq*size^2+1:end])
    state.slater_matrix.inv_m_real[1:nq*size^2].=inverses
    state.slater_matrix.pf_m_real.=pf
    mat_calls[]+=1
    0
end
function sync_shadow!(state)
    state.slater_matrix.inv_m.=ComplexF64.(state.slater_matrix.inv_m_real)
    state.slater_matrix.pf_m.=ComplexF64.(state.slater_matrix.pf_m_real)
end
function real_ip(state,data)
    pf=state.slater_matrix.pf_m_real; weights=data.qp_weights.qp_full_weight
    c=ccall(IPCALL,Float64,(Cint,Ptr{Float64},Ptr{ComplexF64}),length(pf),pf,weights)
    j=MVMCOptimizers.calculate_ip_real(pf,1,length(pf)+1,data)
    compare_stage("public_real_IP",[j],[c]); ComplexF64(j)
end
function public_diff_check(view,ip,idx,spins,data,state)
    n=data.modpara.nsite; nq=length(state.slater_matrix.pf_m); norb=data.modpara.n_orbital_idx
    data.modpara.nsp_gauss_leg==1 && data.n_qp_opt_trans==1 || error("first public differential has no extra spin/opt sectors")
    length(data.qp_trans)>=nq || error("selected QP range exceeds QPTrans header maps")
    orb=Cint[data.orbital_idx_matrix[i,j] for i in 1:2n for j in 1:2n]
    signs=Cint[data.orbital_sgn[i,j] for i in 1:2n for j in 1:2n]
    maps=Cint[v for row in data.qp_trans for v in row]
    mapsgn=Cint[v for row in data.qp_trans_sgn for v in row]
    out=zeros(ComplexF64,2norb); ci=Cint.(idx); cs=Cint.(spins)
    ccall(DIFFCALL,Cvoid,(Cint,Cint,Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Float64,Ptr{ComplexF64}),n,length(idx),nq,norb,orb,signs,maps,mapsgn,ci,cs,state.slater_matrix.inv_m,state.slater_matrix.pf_m,data.qp_weights.qp_full_weight,real(ip),out)
    MVMCOptimizers.slater_elm_diff_fsz!(view,ip,idx,spins,data,state)
    compare_stage("public_original_differential",view,out)
end
function real_accumulate!(acc,o,w,e,size)
    isreal(e) || error("real energy adapter produced imaginary energy")
    even=real.(o[1:2:2size]) # DEFINED C real-driver even-O conversion, not oracle stripping.
    if isempty(C_OO[]); C_OO[]=zeros(length(acc.sr_opt_oo_real)); C_HO[]=zeros(size); end
    ccall(OOCALL,Cvoid,(Ptr{Float64},Ptr{Float64},Ptr{Float64},Float64,Float64,Cint),C_OO[],C_HO[],even,w,real(e),size)
    MVMCOptimizers.calculate_oo_real!(acc.sr_opt_oo_real,acc.sr_opt_ho_real,even,w,real(e),size)
    C_WC[]+=w; measured_samples[]+=1
end
function initialized_checkpoint(data,rng)
    !MVMCOptimizers.get_all_complex_flag(data) || error("initialization changed real mode")
    all(isreal,MVMCOptimizers.pack_parameters(data)) || error("initialized coefficients are not real")
    checkpoint("initialized",data)
    integers(joinpath(DEST,"optimization-flags.txt"),data.optimization_flags)
    data
end
function sampled_checkpoint(data,state,rng)
    checkpoint("sampled",data,state)
end
function pre_sr_checkpoint(data,state,rng)
    C_WC[]>0 || error("zero selected measured samples")
    sr=state.sr_opt
    inverse_weight=1.0/C_WC[] # C average.c computes reciprocal then multiplies.
    compare_stage("full_normalized_preSR_OO",sr.sr_opt_oo_real,C_OO[].*inverse_weight;atol=1e-12,rtol=1e-12)
    compare_stage("full_normalized_preSR_HO",sr.sr_opt_ho_real,C_HO[].*inverse_weight;atol=1e-12,rtol=1e-12)
    PRE_SR[]=deepcopy(sr); SNAPSHOT[]=(deepcopy(data),deepcopy(state))
    checkpoint("pre-sr",data,state)
    numeric(joinpath(DEST,"pre-sr/sr_oo.txt"),sr.sr_opt_oo_real)
    numeric(joinpath(DEST,"pre-sr/sr_ho.txt"),sr.sr_opt_ho_real)
end
function final_checkpoint(data,state,rng)
    checkpoint("final",data,state)
    before=parse(Int,strip(read(joinpath(DEST,"pre-sr/draw-count.txt"),String)))
    draws[]==before || error("SR solver drew RNG")
    read(joinpath(DEST,"final/next624.txt"),String)==read(joinpath(DEST,"pre-sr/next624.txt"),String) || error("SR solver changed RNG state")
end

# Clone the energy adapter without changing the shared source. The actual real
# native library is mandatory, and the public input must select real mode.
adapterpath=joinpath(REPO,"scripts/reference_native_fsz_energy.jl")
adapter=read(adapterpath,String)
bytes2hex(sha256(adapter))=="65451402ef71ccdcae090fce7c258a4c6d400fcfc4604f1e62fa7225b59f4bed" || error("energy adapter source changed")
adapter=replace(adapter,"NativeFSZEnergyReference"=>"Threaded182RealEnergyReference")
adapter=replace_once(adapter,"\"libmvmc_fsz_reference\"","\"libmvmc_fsz_reference_real\"")
adapter=replace_once(adapter,"all_complex || error(\"This runner adapter is scoped to complex FSZ models\")","!all_complex || error(\"Real adaptation requires explicit real input\")")
adapter=replace_once(adapter,"Mixed reference: original Julia 1.13.1 runner/RNG/sampling/SR with actual serial native C FSZ local energy including DH2/DH4; every saved projection counter verified by C MakeProjCnt; no RBM; not full C executable or MPI parity.","ADAPTED C-compatible REAL FSZ reference; driver Mat/IP/shadows/even-O/SR adapted separately by generator; original RNG draws retained. Actual serial native C real local energy; not unmodified Julia, full C executable or MPI parity.")
adapterfile=joinpath(DEST,"clones/real_energy.jl"); write(adapterfile,adapter)
include(adapterfile)
Threaded182RealEnergyReference.install!(abspath(ARGS[3]))

sampler=function_body("vmc_sampling.jl","function vmc_make_sample_fsz_real!(")
sampler=replace_once(sampler,"save_ele_config_fsz!(\n                sample,\n                log_ip_old,","save_ele_config_fsz!(\n                sample,\n                ComplexF64(log_ip_old),")
install_clone("real_sampler_save_boundary",sampler)
install_clone("real_mat", """
function calculate_m_all_fsz_real!(idx::Vector{Int},spins::Vector{Int},qstart::Int,qend::Int,data::ExpertModeData,state::VMCOptimizationState)::Int
    Main.real_mat!(idx,spins,qstart,qend,data,state)
end
""")
main=function_body("vmc_main_cal.jl","function vmc_main_cal_fsz!(")
main=replace_once(main,"function vmc_main_cal_fsz!(","function issue182_real_maincal!(")
selection="    all_complex = true\n    orbital_complex = any(term -> term.is_complex, data.orbital_terms)\n    if data.modpara.complex_flag != 0 || orbital_complex\n        all_complex = true\n    end"
main=replace_once(main,selection,"    all_complex = false\n    get_all_complex_flag(data) && error(\"real-only adapted driver\")\n    (data.modpara.nstore_o == 0 && data.modpara.nsrcg == 0) || error(\"direct/no-store first oracle scope\")")
main=replace_once(main,"info = calculate_m_all_fsz!(ele_idx, ele_spn, 1, n_qp_full + 1, data, worker_state)","info = calculate_m_all_fsz_real!(ele_idx, ele_spn, 1, n_qp_full + 1, data, worker_state)\n        Main.sync_shadow!(worker_state)")
main=replace_once(main,"ip = calculate_ip_fcmp(worker_state.slater_matrix.pf_m, 1, n_qp_full + 1, data; reduce = :none)","ip = Main.real_ip(worker_state,data)")
main=replace_once(main,"slater_elm_diff_fsz!(slater_view, ip, ele_idx, ele_spn, data, worker_state)","Main.public_diff_check(slater_view,ip,ele_idx,ele_spn,data,worker_state)")
a=first(findfirst("            # Accumulate OO and HO ([43] calculate OO and HO)",main))
b=last(findnext("            ctimer_stop!(c_timer, 43)",main,a))
main=main[1:a-1]*"            ctimer_start!(c_timer,43)\n            Main.real_accumulate!(local_acc.sr_opt,sr_opt_o,w,e,sr_opt_size)\n            ctimer_stop!(c_timer,43)"*main[b+1:end]
install_clone("real_maincal",main)
opt=function_body("vmc_para_opt.jl","function vmc_para_opt!(")
opt=replace_once(opt,"function vmc_para_opt!(","function issue182_real_opt!(")
opt=replace_once(opt,"vmc_main_cal_fsz!(data, state, timer, ctx)","issue182_real_maincal!(data, state, timer, ctx)")
opt=replace_once(opt,"        # [4] VMCMainCal","        Main.sampled_checkpoint(data,state,rng)\n        # [4] VMCMainCal")
opt=replace_once(opt,"        # 8. Stochastic optimization","        Main.pre_sr_checkpoint(data,state,rng)\n        # 8. Stochastic optimization")
opt=replace_once(opt,"        # Callback","        Main.final_checkpoint(data,state,rng)\n        # Callback")
install_clone("observed_real_opt",opt)
runner=function_body("run_para_opt_from_namelist.jl","function run_para_opt_from_namelist(")
runner=replace_once(runner,"function run_para_opt_from_namelist(","function issue182_real_runner(")
runner=replace_once(runner,"    init_parameter!(data; rng = rng)","    Main.checkpoint(\"seeded\",data)\n    init_parameter!(data; rng = rng)")
runner=replace_once(runner,"    status = vmc_para_opt!(","    Main.initialized_checkpoint(data,rng)\n    status = issue182_real_opt!(")
install_clone("observed_real_runner",runner)

try
    result=Base.invokelatest(MVMCOptimizers.issue182_real_runner,input;nsteps=1,nsmp=1,mode=:fsz,output_dir=joinpath(DEST,"outputs"),initial_def=:none)
    result.status==0 || error("original real SR solver returned $(result.status)")
    PRE_SR[]!==nothing && measured_samples[]==parsed.modpara.nvmc_sample || error("missing sample/pre-SR coverage")
    bytes2hex(sha256(read(MATLIB)))=="0217b63344df4e0683b9fdd96546c9a6bd571d4feb25fe799ec9b89586dc0478" || error("Mat library changed during oracle")
    bytes2hex(sha256(read(ENERGYLIB)))=="a1906c51587d0b788e1c1763cb1e31a072d946870efb0cc7bcf6be4044a6e131" || error("real-energy library changed during oracle")
    for (name,hash) in GUARDS
        bytes2hex(sha256(read(joinpath(SOURCE,name))))==hash || error("source changed during oracle")
    end
    for (name,hash) in input_hashes
        bytes2hex(sha256(read(joinpath(donor,name))))==hash || error("donor input changed during oracle")
    end
    write(joinpath(DEST,"status.txt"),"0\n")
    open(joinpath(DEST,"provenance.txt"),"w") do io
        println(io,"scope=ADAPTED C-compatible real-FSZ public-derived prefix1 direct/no-store; NOT unmodified Julia or full native C sampler/SR executable")
        println(io,"Julia=$VERSION platform=$(Sys.MACHINE) BLAS=$(BLAS.get_config()) Julia_threads=1 BLAS_threads=1")
        println(io,"input=copy original heisenberg_chain_fsz public definitions; ComplexType1→0; NStore1→0; initial.def deliberately omitted; QPTrans header retained")
        println(io,"real Mat=Julia real PfaPack checked against actual C every call; IP actual C; energy actual real C; original differential with C-checked full real shadows; C even-O real SR; OO/HO original C DGER/DAXPY compared full normalized preSR")
        println(io,"sampler adaptation=unused Float64 save-log argument→ComplexF64 only; no draws/reordering/reseed; original Julia real sampler and SR solver retained")
        println(io,"flags=original C GetInfoOpt/GetInfoOptOrbitalParalell writes detected with two sentinels; unwritten native slots UNKNOWN; inactive zeros explicit defined policy, NOT native malloc values; raw Julia API defaults retained per stage; active real flags exact")
        println(io,"mat_calls=$(mat_calls[]) measured_samples=$(measured_samples[]) words=$(draws[])")
        println(io,"drawcount=actual primitive C gen_rand32/genrand_real2 observer; next624=SFMT save/restore peek; complete saved/burn configurations and counter captured")
        for (name,hash) in sort(collect(GUARDS)); println(io,name," sha256=",hash); end
        for (name,hash) in sort(collect(input_hashes)); println(io,"donor/",name," sha256=",hash); end
        for root in (inputdir,joinpath(DEST,"clones")),name in sort(readdir(root))
            path=joinpath(root,name); println(io,relpath(path,DEST)," sha256=",bytes2hex(sha256(read(path))))
        end
        for path in (MATLIB,ENERGYLIB,cpath,clib,@__FILE__,joinpath(REPO,"extern/Julia-mVMC/Manifest-v1.13.toml"))
            println(io,path," sha256=",bytes2hex(sha256(read(path))))
        end
    end
    println("ORACLE READY FOR REVIEW (not yet fixture acceptance): $DEST")
catch failure
    write(joinpath(DEST,"UNVERIFIED.txt"),sprint(showerror,failure)*"\n")
    rethrow()
finally
    close(LOG)
end
