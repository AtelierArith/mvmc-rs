# Process-local diagnostic; does not replace any public method.
import MVMCOptimizers
Core.eval(MVMCOptimizers, quote
function _issue496_slater_checked_domain(n, ns, nm, no, weights, oi, os, sl,
                                        qt, qsg, qo, qosg, out)
    n > 0 && ns > 0 && nm > 0 && no > 0 || return false
    size(oi) == (n,n) && size(os) == (n,n) || return false
    length(out) >= 4*n*n*ns*nm*no || return false
    all(x -> 0 <= x < length(sl), oi) || return false
    length(weights.spgl_cos_sin) >= ns || return false
    length(weights.spgl_cos_cos) >= ns || return false
    length(weights.spgl_sin_sin) >= ns || return false
    length(qt) >= nm && length(qsg) >= nm || return false
    length(qo) >= no && length(qosg) >= no || return false
    for m in 1:nm
        length(qt[m]) >= n && length(qsg[m]) >= n || return false
        all(x -> 0 <= x < n, qt[m]) || return false
    end
    for o in 1:no
        length(qo[o]) >= n && length(qosg[o]) >= n || return false
        all(x -> 0 <= x < n, qo[o]) || return false
    end
    return true
end

function _issue496_slater_serial_kernel!(out::Vector{ComplexF64}, n::Int,
    ns::Int, nm::Int, no::Int, oi::Matrix{Int}, os::Matrix{Int},
    sl::Vector{ComplexF64}, qt::Vector{Vector{Int}}, qsg::Vector{Vector{Int}},
    qo::Vector{Vector{Int}}, qosg::Vector{Vector{Int}},
    weights::MVMCExpertModeParsers.QuantumProjectionWeights)
    n2=2*n; fix=ns*nm
    @inbounds for q in 1:fix*no
        z=q-1; o=z÷fix+1; rem=z%fix; m=rem÷ns+1; s=rem%ns+1
        xopt=qo[o]; xoptsgn=qosg[o]; x=qt[m]; xsgn=qsg[m]
        cs=weights.spgl_cos_sin[s]; cc=weights.spgl_cos_cos[s]; ss=weights.spgl_sin_sin[s]
        offset=(q-1)*n2*n2
        for ri in 0:n-1
            ori=xopt[ri+1]; tri=x[ori+1]; sgni=xsgn[ori+1]*xoptsgn[ri+1]
            row0=offset+ri*n2; row1=offset+(ri+n)*n2
            for rj in 0:n-1
                orj=xopt[rj+1]; trj=x[orj+1]; sgnj=xsgn[orj+1]*xoptsgn[rj+1]
                ij=oi[tri+1,trj+1]; ji=oi[trj+1,tri+1]
                a=sl[ij+1]*Float64(os[tri+1,trj+1]*sgni*sgnj)
                b=sl[ji+1]*Float64(os[trj+1,tri+1]*sgni*sgnj)
                out[row0+rj+1]=-(a-b)*cs
                out[row0+rj+n+1]=a*cc+b*ss
                out[row1+rj+1]=-a*ss-b*cc
                out[row1+rj+n+1]=(a-b)*cs
            end
        end
    end
    return nothing
end

function _issue496_update_slater_serial_checked!(data::ExpertModeData, state::VMCOptimizationState)
    # Debug mode retains every original log expression and its ordering.
    Base.CoreLogging.min_enabled_level(Base.CoreLogging.current_logger()) <= Base.CoreLogging.Debug &&
        return update_slater_elm_fcmp!(data,state)
    weights=data.qp_weights
    weights isa MVMCExpertModeParsers.QuantumProjectionWeights ||
        return update_slater_elm_fcmp!(data,state)
    n=data.modpara.nsite; ns=max(1,data.modpara.nsp_gauss_leg)
    nm=max(1,data.modpara.nmp_trans); no=max(1,data.n_qp_opt_trans)
    length(weights.spgl_cos) >= ns || return update_slater_elm_fcmp!(data,state)
    # Builders, duplicate/nonzero filtering, and their validation stay unchanged.
    oi,os,sl=build_orbital_idx_sgn_matrices(data)
    qt,_,qsg,qo,qosg=build_qp_trans_matrices(data)
    out=state.slater_matrix.slater_elm
    _issue496_slater_checked_domain(n,ns,nm,no,weights,oi,os,sl,qt,qsg,qo,qosg,out) ||
        return update_slater_elm_fcmp!(data,state)
    return _issue496_slater_serial_kernel!(out,n,ns,nm,no,oi,os,sl,qt,qsg,qo,qosg,weights)
end
end)
