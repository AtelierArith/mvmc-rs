using MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
const MO = MVMCOptimizers
const MP = MVMCExpertModeParsers
BLAS.set_num_threads(1)
# Optional developer diagnostic; no Cargo consumer and no numerical-budget policy.
# The inverse perturbation term uses two OBSERVED input matrices. It is a
# posthoc pairwise explanation, not an admissible future-backend error budget.
repo = normpath(joinpath(@__DIR__, ".."))
root = joinpath(repo, "tests/fixtures/physcal_181/two-samples/heisenberg_chain_fsz")
inverse_input = isempty(ARGS) ? joinpath(repo, "tests/fixtures/physcal_181/native-macos-measurement-diagnostic/mac-inverses.txt") : ARGS[1]
data = MP.parse_expert_mode_files(joinpath(root,"inputs/namelist.def"))
MO.read_opt_para_file!(data, joinpath(root,"zqp_opt.dat"))
MP.read_input_parameters!(data, joinpath(root,"inputs/namelist.def"))
MO.sync_modified_parameter!(MO.serial_context(), data; shift_correlations=false)
data.modpara.nelec == 0 && (data.modpara.nelec = (data.modpara.nlocspin + data.modpara.ncond) ÷ 2)
data.modpara.nmp_trans = max(1,abs(data.modpara.nmp_trans))
ns = data.modpara.nsite; ne = data.modpara.nelec; n = 2ne
state = MO.VMCOptimizationState(ns,ne,MP.projection_layout(data).n_proj,
    MP.count_variational_parameters(data),1,200,true,true)
MP.init_qp_weight!(data)
MO.update_slater_elm_fsz!(data,state)
println("REFERENCE Julia=", VERSION," module=",pathof(MO)," BLAS=", BLAS.get_config())
empty = deepcopy(data)
families = (:coulomb_intra_terms,:coulomb_inter_terms,:hund_terms,:transfer_terms,:pair_hop_terms,:exchange_terms,:inter_all_terms)
for field in families; empty!(getproperty(empty,field)); end
function polynomial(inverse,a,b,ma,mb)
    N=length(a)
    pa=pb=qa=qb=zero(eltype(inverse))
    for i=1:N
        pa+=inverse[(ma-1)*N+i]*a[i]; pb+=inverse[(mb-1)*N+i]*a[i]
        qa+=inverse[(ma-1)*N+i]*b[i]; qb+=inverse[(mb-1)*N+i]*b[i]
    end
    bma=zero(eltype(inverse))
    for i=1:N
        tmp=zero(eltype(inverse))
        for j=1:N; tmp+=inverse[(i-1)*N+j]*a[j]; end
        bma+=b[i]*tmp
    end
    ab=inverse[(ma-1)*N+mb]
    terms=[ab*b[ma],ab*bma,pa*qb,-pb*qa]
    ratio=terms[1]+terms[2]+terms[3]+terms[4]
    return (;pa,pb,qa,qb,bma,ab,terms,ratio)
end
function observed_input_change_bound(inv,a,b,ma,mb,change)
    N=length(a); p=polynomial(inv,a,b,ma,mb)
    dp=sum(abs(change[(ma-1)*N+i])*abs(a[i]) for i=1:N)
    dq=sum(abs(change[(ma-1)*N+i])*abs(b[i]) for i=1:N)
    ep=sum(abs(change[(mb-1)*N+i])*abs(a[i]) for i=1:N)
    eq=sum(abs(change[(mb-1)*N+i])*abs(b[i]) for i=1:N)
    db=sum(abs(b[i])*abs(change[(i-1)*N+j])*abs(a[j]) for i=1:N,j=1:N)
    dab=abs(change[(ma-1)*N+mb])
    return dab*abs(b[ma])+dab*abs(p.bma)+abs(p.ab)*db+dab*db+
           dp*abs(p.qb)+abs(p.pa)*eq+dp*eq+ep*abs(p.qa)+abs(p.pb)*dq+ep*dq
end

function arithmetic_bound(input,a,b,ma,mb,ip)
    N=length(a); u=BigFloat(2)^(-53); gamma2=2u/(1-2u)
    ball(z)=(value=z,radius=BigFloat(0))
    function add(x,y)
        value=x.value+y.value
        propagated=x.radius+y.radius
        return (value=value,radius=propagated+u/(1-u)*(abs(value)+propagated))
    end
    function multiply(x,y)
        value=x.value*y.value
        propagated=abs(x.value)*y.radius+abs(y.value)*x.radius+x.radius*y.radius
        # A complex multiplication's real/imag components each have two
        # scalar products and one addition: each product's path has two
        # roundings, hence gamma_2 times that component's absolute-product sum.
        re_scale=abs(real(x.value)*real(y.value))+abs(imag(x.value)*imag(y.value))
        im_scale=abs(real(x.value)*imag(y.value))+abs(imag(x.value)*real(y.value))
        return (value=value,radius=propagated+gamma2*(hypot(re_scale,im_scale)+2propagated))
    end
    pa=pb=qa=qb=ball(zero(eltype(input)))
    for i=1:N
        ia=ball(input[(ma-1)*N+i]); ib=ball(input[(mb-1)*N+i])
        pa=add(pa,multiply(ia,ball(a[i]))); pb=add(pb,multiply(ib,ball(a[i])))
        qa=add(qa,multiply(ia,ball(b[i]))); qb=add(qb,multiply(ib,ball(b[i])))
    end
    bma=ball(zero(eltype(input)))
    for i=1:N
        tmp=ball(zero(eltype(input)))
        for j=1:N; tmp=add(tmp,multiply(ball(input[(i-1)*N+j]),ball(a[j]))); end
        bma=add(bma,multiply(ball(b[i]),tmp))
    end
    ab=ball(input[(ma-1)*N+mb])
    ratio=add(add(add(multiply(ab,ball(b[ma])),multiply(ab,bma)),multiply(pa,qb)),
              (value=-multiply(pb,qa).value,radius=multiply(pb,qa).radius))
    # The following finite normal-range scalar balls follow the actual
    # quotient paths. No invented fixed operation-count tail is used.
    function checked(value,radius)
        @assert isfinite(value) && isfinite(radius) && radius>=0
        magnitude=abs(value)+radius
        @assert magnitude<BigFloat(floatmax(Float64))
        @assert value==0 || abs(value)>=BigFloat(floatmin(Float64))
        return (value=value,radius=radius)
    end
    scalar(value,radius=BigFloat(0))=checked(BigFloat(value),radius)
    function sadd(x,y)
        value=x.value+y.value; propagated=x.radius+y.radius
        checked(value,propagated+u/(1-u)*(abs(value)+propagated))
    end
    function smul(x,y)
        value=x.value*y.value
        propagated=abs(x.value)*y.radius+abs(y.value)*x.radius+x.radius*y.radius
        checked(value,propagated+u/(1-u)*(abs(value)+propagated))
    end
    function sfma(x,y,z)
        value=x.value*y.value+z.value
        propagated=abs(x.value)*y.radius+abs(y.value)*x.radius+x.radius*y.radius+z.radius
        checked(value,propagated+u/(1-u)*(abs(value)+propagated))
    end
    function sdiv(x,y)
        @assert abs(y.value)>y.radius # safe denominator excludes zero
        value=x.value/y.value
        propagated=(x.radius+abs(value)*y.radius)/(abs(y.value)-y.radius)
        checked(value,propagated+u/(1-u)*(abs(value)+propagated))
    end
    negate(x)=scalar(-x.value,x.radius)
    # PfMNew=ratio*PfM, then CalculateIP's single QP multiply/add, then
    # projection factor. This input has exact QP weight=1 and projection=1.
    numerator=multiply(multiply(multiply(ratio,ball(Complex{BigFloat}(ip))),
                                ball(one(eltype(input)))),ball(one(eltype(input))))
    ar=scalar(real(numerator.value),numerator.radius)
    ai=scalar(imag(numerator.value),numerator.radius)
    cr=scalar(real(ip)); ci=scalar(imag(ip))
    # Native macOS ARM C __divdc3: exact exponent scaling of denominator,
    # d*d, fused c*c+d*d, fused numerator, division, exact binary unscale.
    power=frexp(max(abs(real(ip)),abs(imag(ip))))[2]-1
    factor=BigFloat(2)^(-power)
    c=scalar(cr.value*factor); d=scalar(ci.value*factor)
    dd=smul(d,d); denominator=sfma(c,c,dd)
    @assert denominator.value-denominator.radius>0
    re=sdiv(sfma(ar,c,smul(ai,d)),denominator)
    im=sdiv(sfma(ai,c,negate(smul(ar,d))),denominator)
    mac_radius=hypot(re.radius,im.radius)*factor
    # Julia's safe ordinary-range Smith path, Base/complex.jl:
    # r=d/c; t=1/(c+d*r); (a+b*r)*t and (b-a*r)*t.
    # Captured IP operands satisfy nonzero r and nonzero numerator*r,
    # so neither fallback component branch nor extreme-range scaling applies.
    @assert max(abs(cr.value),abs(ci.value))>BigFloat(floatmin(Float64))*2/eps(Float64)
    @assert max(abs(cr.value),abs(ci.value))<BigFloat(floatmax(Float64))/2
    a,b,c,d,swapped = abs(ci.value)<=abs(cr.value) ? (ar,ai,cr,ci,false) : (ai,ar,ci,cr,true)
    r=sdiv(d,c); @assert r.value!=0
    t=sdiv(scalar(1),sadd(c,smul(d,r)))
    br=smul(b,r); nar=smul(negate(a),r)
    @assert br.value!=0 && nar.value!=0
    re=smul(sadd(a,br),t); im=smul(sadd(b,nar),t)
    julia_radius=hypot(re.radius,im.radius)
    return max(mac_radius,julia_radius)

end
mac_rows=Dict{Tuple{Int,Int},Vector{ComplexF64}}()
for line in eachline(inverse_input)
    fields=split(line); @assert length(fields)==74
    values=parse.(Float64,fields[3:end])
    mac_rows[(parse(Int,fields[1]),parse(Int,fields[2]))]=complex.(values[1:2:end],values[2:2:end])
end
@assert length(mac_rows)==400
setprecision(256) do
    frame=0; stage=joinpath(root,"sample-$frame")
    records=Dict(field=>parse.(Int,split(read(joinpath(stage,"$field.txt"),String))) for field in (:ele_idx,:ele_cfg,:ele_num,:ele_proj_cnt,:ele_spn))
    sums=ComplexF64[0,0]; term_sum_bound=BigFloat(0); sum_abs=BigFloat[0,0]
    largest=(BigFloat(0),-1); first=-1
    for walker=0:199
        idx=records[:ele_idx][walker*n+1:(walker+1)*n]; spins=records[:ele_spn][walker*n+1:(walker+1)*n]
        cfg=records[:ele_cfg][walker*2ns+1:(walker+1)*2ns]; num=records[:ele_num][walker*2ns+1:(walker+1)*2ns]
        np=MP.projection_layout(data).n_proj; cnt=records[:ele_proj_cnt][walker*np+1:(walker+1)*np]
        @assert MO.calculate_m_all_fsz!(idx,spins,1,2,data,state)==0
        ip=state.slater_matrix.pf_m[1]; @assert data.qp_weights.qp_full_weight==[1.0+0.0im]; linux=copy(state.slater_matrix.inv_m[1:n*n]); mac=mac_rows[(0,walker)]
        values=ComplexF64[]
        for input in (linux,mac)
            state.slater_matrix.inv_m[1:n*n]=input
            push!(values,MO.green_func2_fsz(0,1,1,0,0,1,ip,idx,cfg,num,cnt,spins,data,state;all_complex=true))
        end
        sums.+=values; sum_abs.+=abs.(Complex{BigFloat}.(values))
        if values[1]!=values[2] && first<0; first=walker; end
        ri=0;rj=1;rk=1;rl=0;s=0;t=1
        if num[ri+s*ns+1]==1 || num[rj+s*ns+1]==0 || num[rk+t*ns+1]==1 || num[rl+t*ns+1]==0
            @assert values==zeros(ComplexF64,2); continue
        end
        ma=cfg[rl+t*ns+1]+1; mb=cfg[rj+s*ns+1]+1
        moved_idx=copy(idx); moved_spins=copy(spins)
        moved_idx[ma]=rk; moved_spins[ma]=t; moved_idx[mb]=ri; moved_spins[mb]=s
        moved_num=copy(num); cnt1=zeros(Int,np); cnt2=zeros(Int,np)
        moved_num[rl+t*ns+1]=0; moved_num[rk+t*ns+1]=1
        MO.update_proj_cnt!(rl,rk,t,cnt1,cnt,moved_num,data)
        moved_num[rj+s*ns+1]=0; moved_num[ri+s*ns+1]=1
        MO.update_proj_cnt!(rj,ri,s,cnt2,cnt1,moved_num,data)
        @assert MO.proj_rbm_ratio(cnt2,cnt,moved_num,num,data)==1.0+0.0im

        rsa=moved_idx[ma]+t*ns; rsb=moved_idx[mb]+s*ns
        a=Complex{BigFloat}.([state.slater_matrix.slater_elm[rsa*2ns+moved_idx[i]+moved_spins[i]*ns+1] for i=1:n])
        b=Complex{BigFloat}.([state.slater_matrix.slater_elm[rsb*2ns+moved_idx[i]+moved_spins[i]*ns+1] for i=1:n])
        highlinux=Complex{BigFloat}.(linux); highmac=Complex{BigFloat}.(mac)
        pl=polynomial(highlinux,a,b,ma,mb); pm=polynomial(highmac,a,b,ma,mb)
        inv_change=highmac-highlinux
        for k=1:n; inv_change[(k-1)*n+k]=0; end
        offdiag_delta=abs(polynomial(highlinux+inv_change,a,b,ma,mb).ratio-pl.ratio)
        bound=observed_input_change_bound(highlinux,a,b,ma,mb,highmac-highlinux)+
              arithmetic_bound(highlinux,a,b,ma,mb,ip)+arithmetic_bound(highmac,a,b,ma,mb,ip)
        @assert abs(values[2]-values[1])<=bound
        term_sum_bound+=bound
        actual_error=abs(values[2]-values[1])
        if actual_error>largest[1]; largest=(actual_error,walker); end
        A=Complex{BigFloat}.([-state.slater_matrix.slater_elm[(idx[i]+spins[i]*ns)*2ns+idx[j]+spins[j]*ns+1] for j=1:n,i=1:n])
        residual=opnorm(A*(-reshape(highmac,n,n))-I,Inf)
        backward=residual/(opnorm(A,Inf)*opnorm(reshape(highmac,n,n),Inf)+1)
        println("DIRECT13 frame=0 walker=$walker scope=posthoc-diagnostic weight=1 linux=$(repr(values[1])) mac_inverse_replay=$(repr(values[2])) terms=$(repr(pl.terms)) cancellation=$(sum(abs,pl.terms)/abs(pl.ratio)) offdiag_exact_input_delta=$offdiag_delta observed_input_change_bound=$(observed_input_change_bound(highlinux,a,b,ma,mb,highmac-highlinux)) pairwise_forward_bound=$bound condInf=$(cond(ComplexF64.(A),Inf)) mac_inverse_backward=$backward mac_residual=$residual")
    end
    u=BigFloat(2)^(-53); gamma=k->k*u/(1-k*u)
    sum_bound=term_sum_bound+gamma(200)*(sum_abs[1]+sum_abs[2])
    println("FRAME0_DIRECT13 first_different_walker=$first largest_local_error=$(largest[1]) largest_walker=$(largest[2]) weight_per_walker=1 accumulated_weight=200 linux_sum=$(repr(sums[1])) mac_inverse_replay_sum=$(repr(sums[2])) term_bound_sum=$term_sum_bound summation_bound=$sum_bound sum_abs_linux=$(sum_abs[1]) sum_abs_mac=$(sum_abs[2])")
end


