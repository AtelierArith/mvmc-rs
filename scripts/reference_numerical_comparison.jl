# Computed binary64 values are compared semantically; encoded inputs and
# deterministic controls must be classified explicitly by the caller.
module ReferenceNumericalComparison
using LinearAlgebra
export compare_hex_text, compare_unused_snapshot, compare_computed_text, compare_runner, compare_cg_fixed, compare_initialization_text, compare_sampling_text, compare_record_blocks, within

function within(a, e, atol, rtol)
    isfinite(atol) && atol >= 0 && isfinite(rtol) && rtol >= 0 || error("Invalid numerical bounds")
    (isnan(a) || isnan(e)) && return isnan(a) && isnan(e)
    (isinf(a) || isinf(e)) && return a == e
    scale=max(abs(a),abs(e))
    difference=abs(a-e)
    !isfinite(difference) && return abs(a/scale-e/scale)<=atol/scale+rtol
    return difference <= atol + rtol*scale
end
decode(word) = occursin(r"^[0-9a-fA-F]{16}$", word) ? reinterpret(Float64, parse(UInt64, word; base=16)) : error("Invalid binary64 token: $word")
values(row) = decode.(split(row))
# The BLAS library name/ABI is recorded as provenance, but source versions,
# threads, seeds and kernel SHA tokens remain exact.
header_contract(line) = replace(line,r"; LBTConfig\([^;]*\)"=>"; <recorded BLAS provider>")
function close_values(actual, expected, atol, rtol; context="computed values")
    length(actual) == length(expected) || error("$context: changed cardinality")
    for (i,(a,e)) in enumerate(zip(actual,expected))
        within(a,e,atol,rtol) || error("$context[$i]: $a != $e (absolute=$atol relative=$rtol)")
    end
    return true
end
function compare_hex_text(actual, expected, selector;count_empty=false)
    aa, ee = split(actual,'\n';keepempty=true), split(expected,'\n';keepempty=true)
    length(aa) == length(ee) || error("Changed reference line count")
    row = 0
    for (line,(a,e)) in enumerate(zip(aa,ee))
        if isempty(strip(e)) || startswith(strip(e),"#")
            header_contract(a) == header_contract(e) || error("Changed reference header/structure at line $line")
            isempty(strip(e)) && count_empty && (row+=1)
            continue
        end
        row += 1
        av,ev = split(a),split(e)
        length(av) == length(ev) || error("Changed token count at line $line")
        for (column,(x,y)) in enumerate(zip(av,ev))
            bounds = selector(row,column,ev)
            if bounds === nothing
                x == y || error("Changed exact field at line $line column $column")
            else
                close_values([decode(x)],[decode(y)],bounds...;context="line $line column $column")
            end
        end
    end
    return true
end
function compare_record_blocks(actual,expected,stride,computed;atol=1e-13,rtol=1e-13,preserve_sentinels=false,preserve_tiny=false)
    selector=(r,c,t)->begin
        mod1(r,stride) in computed || return nothing
        # Partial-view derivative probes intentionally leave these slots intact.
        preserve_sentinels && decode(t[c]) in (7.0,-9.0) && return nothing
        if preserve_tiny && 0<abs(decode(t[c]))<sqrt(floatmin(Float64))
            return (min(atol,4*nextfloat(0.0)),min(rtol,64*eps(Float64)))
        end
        return (atol,rtol)
    end
    return compare_hex_text(actual,expected,selector;count_empty=true)
end
function compare_computed_text(actual,expected;atol=1e-13,rtol=1e-13,zero_pattern=false,selector=nothing)
    fields=(r,c,t)->selector===nothing ? (occursin(r"^[0-9a-fA-F]{16}$",t[c]) ? (atol,rtol) : nothing) : (selector(r,c,t) ? (atol,rtol) : nothing)
    if zero_pattern
        a,e=data_rows(actual),data_rows(expected)
        length(a)==length(e) || error("Changed zero-pattern structure")
        for (row,(ar,er)) in enumerate(zip(a,e))
            av,ev=split(ar),split(er)
            length(av)==length(ev) || error("Changed zero-pattern cardinality")
            for (column,(x,y)) in enumerate(zip(av,ev))
                fields(row,column,ev)===nothing && continue
                iszero(decode(x))==iszero(decode(y)) || error("Changed cutoff zero/nonzero decision")
            end
        end
    end
    return compare_hex_text(actual,expected,fields)
end

function compare_runner(name,actual,expected;sample_count=2000)
    if name in ("fixed-input.txt","gram.txt")
        return compare_unused_snapshot(name,actual,expected;sample_count)
    elseif occursin(r"(?:rng|configs|status)\.txt$",name) || occursin("flags",name)
        actual==expected || error("Changed exact runner controls: $name")
        return true
    elseif name=="initial-parameters.txt"
        return compare_computed_text(actual,expected;atol=32*eps(Float64),rtol=32*eps(Float64))
    elseif endswith(name,"parameters.txt") || endswith(name,"energy.txt")
        return compare_computed_text(actual,expected;atol=1e-11,rtol=1e-11)
    elseif endswith(name,".dat") || occursin("srinfo",lowercase(name))
        aa,ee=split(actual,'\n';keepempty=true),split(expected,'\n';keepempty=true)
        length(aa)==length(ee) || error("Changed runner output length: $name")
        for (line,(a,e)) in enumerate(zip(aa,ee))
            if isempty(strip(e)) || startswith(strip(e),"#")
                a==e || error("Changed runner text/header: $name:$line")
                continue
            end
            av,ev=split(a),split(e)
            length(av)==length(ev) || error("Changed runner output cardinality: $name:$line")
            srinfo=occursin("srinfo",lowercase(name))
            indexed=any(endswith(name,suffix) for suffix in ("zqp_gutzwiller_opt.dat","zqp_jastrow_opt.dat","zqp_orbital_opt.dat"))
            var=endswith(name,"zvo_var.dat")
            # Named header records contain literal dimensions, even if a future
            # serializer chooses decimal formatting for their integer values.
            header=any(occursin(r"[A-DF-Za-df-z_=]",token) for token in ev)
            for (column,(x,y)) in enumerate(zip(av,ev))
                discrete=header || (srinfo && column in (1,2,3,4,8,9)) || (indexed && column==1) || (var && column%3==0)
                if !discrete && occursin(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+[eE][+-]?\d+)(?:[eE][+-]?\d+)?$",y)
                    rtol=srinfo ? 1e-5 : 1e-11
                    atol=srinfo ? 1e-12 : 1e-11
                    within(parse(Float64,x),parse(Float64,y),atol,rtol) || error("Changed runner numerical output: $name:$line:$column")
                else
                    x==y || error("Changed runner index/iteration/text: $name:$line:$column")
                end
            end
        end
        return true
    end
    # Reference metadata and unknown kinds are never silently made numeric.
    header_contract(actual)==header_contract(expected) || error("Changed runner metadata: $name")
    return true
end
data_rows(text) = filter(l -> !isempty(strip(l)) && !startswith(strip(l),"#"), split(text,'\n'))
function sfmt_rows(expected)
    rows=filter(l->!startswith(strip(l),"#"),split(chomp(expected),'\n'))
    return findall(row->begin
        tokens=split(row)
        length(tokens)==624 && all(occursin(r"^\d{1,10}$",t) && parse(UInt64,t)<=typemax(UInt32) for t in tokens)
    end,rows)
end
function compare_initialization_text(actual,expected)
    # These initializer schemas put the computed parameter row immediately
    # before each complete 624-word SFMT block. Loaded/copied rows stay exact.
    computed=Set(sfmt_rows(expected).-1)
    return compare_hex_text(actual,expected,(r,c,t)->r in computed ? (32*eps(Float64),32*eps(Float64)) : nothing;count_empty=true)
end
function compare_sampling_text(actual,expected)
    # Each setup/sampling boundary publishes computed Pfaffians and inverse
    # rows, then 624 UInt32 words. All preceding configuration/control rows
    # (including any encoded 64-bit metadata) retain exact comparison.
    blocks=sfmt_rows(expected)
    computed=Set(vcat(blocks.-1,blocks.-2))
    return compare_hex_text(actual,expected,(r,c,t)->r in computed ? (1e-13,1e-13) : nothing;count_empty=true)
end

function compare_cg_fixed(actual,expected)
    a,e=(filter(l->!startswith(strip(l),"#"),split(chomp(text),'\n')) for text in (actual,expected))
    n,samples,complex=parse.(Int,split(e[1]))
    length(e)==130 && length(a)==130 || error("Unknown fixed CG schema")
    operator_budget=8*(n+samples)*eps(Float64)
    function selector(row,column,tokens)
        complex==0 && row>=5 && (row+=1) # the real-only imaginary sample row is empty
        row==3 && return (operator_budget,operator_budget) # sampled diagonal reduction
        row<=6 && return nothing # means, raw samples and gradient are inputs
        row==7 && return (operator_budget,operator_budget)
        offset=row-8
        offset%3==0 && column<=2 && return nothing # iteration limit/count
        limit=offset÷3+1
        budget=8*(n+samples)*limit*eps(Float64)
        return (budget,budget)
    end
    compare_hex_text(actual,expected,selector)
    mean=values(e[2]); diagonal=values(e[3]); g=values(e[6])
    re=reshape(values(e[4]),n,samples); im=complex==1 ? reshape(values(e[5]),n,samples) : zeros(n,samples)
    # Independent explicit covariance and backward residual, rather than a
    # tolerance derived only from the solver's returned vector.
    S=(re*re'+(complex==1 ? im*im' : zeros(n,n)))/samples-mean*mean'
    S[diagind(S)] .+= 1e-5 .* diagonal
    for limit in 1:41
        index=8+3*(limit-1)
        x=decode.(split(a[index])[3:end]); residual=values(a[index+1])
        scale=abs.(g)+abs.(S)*abs.(x)
        explicit=g-S*x
        budget=16*(n+samples)*limit*eps(Float64)
        all(within(r,q,budget*s,0.0) for (r,q,s) in zip(residual,explicit,scale)) || error("CG limit $limit explicit residual check failed")
    end
    return true
end

function compare_unused_snapshot(name, actual, expected; sample_count=2000)
    sample_count >= 1 || error("Invalid reduction operation count")
    a,e = data_rows(actual),data_rows(expected)
    if name == "gram.txt"
        length(e) == 3 || error("Unknown Gram snapshot schema")
        dims = parse.(Int,split(e[1])); length(dims)==2 || error("Invalid Gram dimensions")
        sample_count=dims[2]
        # Each Gram component is a sum of sample_count products; 64 rounding
        # units per product/reduction account for complex accumulation paths.
        bound=64*sample_count*eps(Float64)
        scale=max(1.0,maximum(abs,values(e[3])))
        return compare_hex_text(actual,expected,(r,c,t)->r==1 ? nothing : r==2 ? (1e-12,1e-12) : (bound*scale,bound))
    end
    name == "fixed-input.txt" || error("Unknown unused snapshot kind: $name")
    length(e)==8 && length(a)==8 || error("Unknown Direct fixed-input schema")
    dims=parse.(Int,split(e[1])); length(dims)==2 || error("Invalid Direct dimensions")
    n=dims[2]
    n > 0 || error("Empty Direct system")
    S=reshape(values(e[5]),n,n); Sa=reshape(values(a[5]),n,n)
    g=values(e[6]); ga=values(a[6]); x=values(e[8]); xa=values(a[8])
    # Reject changed inputs before deriving a sensitivity allowance.
    # Rows 7/8 require a condition-aware forward allowance and independent
    # backward-error checks. Do not use an unconditional runner tolerance.
    kappa=cond(S,Inf)
    isfinite(kappa) && kappa*64*n*eps(Float64)<0.1 || error("Direct reference too ill-conditioned for forward comparison: $kappa")
    # Conditioning is a diagnostic and the residual is an independent gate;
    # it does not grant a blanket kappa*epsilon forward allowance.
    compare_hex_text(actual,expected,(r,c,t)->r<=2 ? nothing : r<=6 ? (1e-12,1e-12) : (1e-11,1e-11))
    backward(S,g,x)=norm(S*x-g,Inf)/max(opnorm(S,Inf)*norm(x,Inf)+norm(g,Inf),floatmin(Float64))
    backward(Sa,ga,xa)<=max(4*backward(S,g,x),128*n*eps(Float64)) || error("Direct solution backward error increased")
    U=Matrix(UpperTriangular(reshape(values(a[7]),n,n)))
    norm(U'*U-Sa,Inf)<=256*n*eps(Float64)*max(1.0,norm(Sa,Inf)) || error("Cholesky factor reconstruction failed")
    return true
end
end
