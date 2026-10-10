from pathlib import Path
import json, math, statistics, sys, hashlib
root=Path(__file__).parent
atol=rtol=64*sys.float_info.epsilon

def number_lines(path):
    metadata=[]; rows=[]
    for line in path.read_text().splitlines():
        if line.startswith(("#", "====", "NGutzwillerIdx", "NJastrowIdx", "NOrbitalIdx")) or not line.strip(): metadata.append(line)
        else: rows.append([float(v) for v in line.split()])
    return metadata,rows

def compare_outputs(a,b):
    def names(p): return sorted(f.name for f in p.glob("*.dat") if not f.name.startswith("zvo_time_"))
    na=names(a); assert na and na==names(b),(a,b)
    count=0; maximum=0.0
    for name in na:
        ma,ra=number_lines(a/name); mb,rb=number_lines(b/name)
        assert ma==mb and len(ra)==len(rb),(a,b,name,"shape/header")
        for row,(xs,ys) in enumerate(zip(ra,rb)):
            assert len(xs)==len(ys),(name,row)
            for col,(x,y) in enumerate(zip(xs,ys)):
                assert math.isfinite(x) and math.isfinite(y),(name,row,col,x,y)
                delta=abs(x-y)
                assert delta<=atol+rtol*max(abs(x),abs(y)),(a,b,name,row,col,x,y,delta)
                maximum=max(maximum,delta); count+=1
    return {"files":len(na),"values":count,"max_abs_delta":maximum}

def timing(d,ranks,threads):
    log=(d/"run.log").read_text().splitlines()
    for rank in range(ranks):
        for value in (f"WORLD {rank} {ranks}",f"THREADS {rank} {threads}",f"BLAS_THREADS {rank} 1"):
            assert log.count(value)==1,(d,value)
    values=[float(l.split()[2]) for l in log if l.startswith("BENCH ")]
    assert len(values)==3 and all(math.isfinite(v) and v>0 for v in values),(d,values)
    return {"times_seconds":values,"median_seconds":statistics.median(values),"binary_sha256":(d/"binary-sha256.txt").read_text().split()[0]}

cases=[]
for size in (32,64):
    for mode in ("opt","phys"):
        if mode=="opt": paths={v:root/f"Rust-bounded-{v}-L{size}-r1-t16" for v in ("baseline","baseline-after")}
        else: paths={v:root/f"Rust-measurement-phys-{v}-L{size}-r1-t16" for v in ("baseline","baseline-after")}
        paths["measurement"]=root/f"Rust-measurement-final-{mode}-L{size}-r1-t16"
        observations={v:timing(p,1,16) for v,p in paths.items()}
        checks=[]
        for version in ("measurement","baseline-after"):
            for rep in range(1,4): checks.append(compare_outputs(paths["baseline"]/f"run-{rep}",paths[version]/f"run-{rep}"))
        cases.append({"mode":mode,"sites":size,"ranks":1,"threads":16,"measurement_team_max_budget":8,"timings":observations,"improvement_percent":{v:100*(1-observations["measurement"]["median_seconds"]/observations[v]["median_seconds"]) for v in ("baseline","baseline-after")},"checks":checks})
for ranks,threads in ((1,1),(2,8),(4,4),(8,2),(16,1)):
    for size in (32,64):
        for mode in ("opt","phys"):
            paths={v:root/f"Rust-measurement-regression-{mode}-{v}-L{size}-r{ranks}-t{threads}" for v in ("baseline","measurement")}
            after=root/f"Rust-measurement-regression-{mode}-baseline-after-L{size}-r{ranks}-t{threads}"
            if after.exists(): paths["baseline-after"]=after
            observations={v:timing(p,ranks,threads) for v,p in paths.items()}
            checks=[compare_outputs(paths["baseline"]/f"run-{rep}",paths["measurement"]/f"run-{rep}") for rep in range(1,4)]
            if "baseline-after" in paths:
                checks += [compare_outputs(paths["baseline"]/f"run-{rep}",paths["baseline-after"]/f"run-{rep}") for rep in range(1,4)]
            cases.append({"mode":mode,"sites":size,"ranks":ranks,"threads":threads,"measurement_team_max_budget":None,"timings":observations,"change_percent":100*(observations["measurement"]["median_seconds"]/observations["baseline"]["median_seconds"]-1),"checks":checks})
(root/"rust-measurement-final-summary.json").write_text(json.dumps({"atol":atol,"rtol":rtol,"warmups":1,"reps":3,"total_samples":320,"cases":cases},indent=2)+"\n")
for case in cases: print(case["mode"],case["sites"],case["ranks"],case["threads"], {v:round(x["median_seconds"],6) for v,x in case["timings"].items()}, max(x["max_abs_delta"] for x in case["checks"]))
audits=[]
for ranks,threads in ((1,16),(4,4)):
    for size in (32,64):
        for steps in (20,300):
            paths={v:root/f"Rust-measurement-rng-{v}-L{size}-r{ranks}-t{threads}-s{steps}" for v in ("baseline","measurement")}
            checks=[]
            for rank in range(ranks):
                name=f"rng-rank-{rank}.json"
                a=json.loads((paths["baseline"]/name).read_text()); b=json.loads((paths["measurement"]/name).read_text())
                x=a.pop("final_energy_per_site"); y=b.pop("final_energy_per_site")
                assert a==b,(size,ranks,threads,steps,rank,"exact RNG/control")
                assert len(a["rng_words"])==624 and a["rank"]==rank and a["world_size"]==ranks and a["steps"]==steps
                if rank==0:
                    assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=atol+rtol*max(abs(x),abs(y))
                    delta=abs(x-y)
                else: assert x is None and y is None; delta=None
                checks.append({"rank":rank,"exact_sfmt_and_discrete_state":True,"rng_words_consumed":a["rng_words_consumed"],"final_energy_abs_delta":delta})
            production=compare_outputs(paths["baseline"]/"production",paths["measurement"]/"production")
            evidence={v:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[v].glob("rng-rank-*.json")} for v in paths}
            audits.append({"sites":size,"ranks":ranks,"threads":threads,"steps":steps,"checks":checks,"production_output":production,"evidence_sha256":evidence,"baseline_binary_sha256":(paths["baseline"]/"binary-sha256.txt").read_text().split()[0],"candidate_binary_sha256":(paths["measurement"]/"binary-sha256.txt").read_text().split()[0]})
(root/"rust-measurement-final-audit-summary.json").write_text(json.dumps({"baseline_source":"370cf7a38fde1632757e73992ce8ffc99119bfe0","atol":atol,"rtol":rtol,"cases":audits},indent=2)+"\n")
print("AUDIT",len(audits),"cases",sum(len(c["checks"]) for c in audits),"paired ranks exact")
