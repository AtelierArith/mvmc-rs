#!/usr/bin/env python3
"""Assemble the FSZ (general-orbital) + DH / RBM / OptTrans input sets (issue #397).

Base: the 6-site Hubbard FSZ definition of Julia's `hubbard_chain_pairhop_fsz`
(Orbital + OrbitalParallel => iFlgOrbitalGeneral = 1; PairHop term dropped). The DH2/DH4
index and `In*` overlay files, the OptTrans definition/overlay and the nine RBM
sections are the 6-site files of `hubbard_chain_dh_rbm_opttrans`. Each set is written to
`tests/fixtures/native_c_physcal_181/_sources/<name>/inputs`; its `zqp_opt.dat` is a
placeholder that `generate.py` replaces through the `zqp=c_opt` variant (one native-C
optimization step), so the fixed parameters never come from Rust.

Run from the repository root with plain python3.
"""
import os, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "extern/Julia-mVMC/test/integration/reference/hubbard_chain_pairhop_fsz/inputs")
DONOR = os.path.join(ROOT, "tests/fixtures/native_c_physcal_181/hubbard_chain_dh_rbm_opttrans/inputs")
OUT = os.path.join(ROOT, "tests/fixtures/native_c_physcal_181/_sources")

RBM = [f"{kind}RBM_{layer}" for kind in ("Charge", "Spin", "General")
       for layer in ("PhysLayer", "HiddenLayer", "PhysHidden")]
COMMON = ["coulombintra.def", "greenone.def", "greentwo.def", "gutzwilleridx.def", "jastrowidx.def",
          "locspn.def", "orbitalidx.def", "orbitalidxpara.def", "qptransidx.def", "trans.def"]
SETS = {
    "fsz_dh2": ["dh2"],
    "fsz_dh24": ["dh2", "dh4"],
    "fsz_dh24_opttrans": ["dh2", "dh4", "opttrans"],
    "fsz_rbm": ["rbm"],
    "fsz_dh24_rbm_opttrans": ["dh2", "dh4", "rbm", "opttrans"],
}

for name, parts in SETS.items():
    dest = os.path.join(OUT, name)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.join(dest, "inputs"))
    for f in COMMON:
        shutil.copy(os.path.join(BASE, f), os.path.join(dest, "inputs", f))
    modpara = open(os.path.join(BASE, "modpara.def")).read()
    if "rbm" in parts:  # hidden-layer neuron counts of the donor RBM definitions
        modpara = modpara.rstrip("\n") + "\nNneuronCharge 2\nNneuronSpin 2\nNneuronGeneral 2\n"
    open(os.path.join(dest, "inputs", "modpara.def"), "w").write(modpara)
    lines = [
        "ModPara modpara.def", "LocSpin locspn.def", "Trans trans.def",
        "CoulombIntra coulombintra.def", "OneBodyG greenone.def", "TwoBodyG greentwo.def",
        "Gutzwiller gutzwilleridx.def", "Jastrow jastrowidx.def",
    ]
    extra = []
    for part in parts:
        if part in ("dh2", "dh4"):
            for f in (f"{part}.def", f"in{part}.def"):
                shutil.copy(os.path.join(DONOR, f), os.path.join(dest, "inputs", f))
            lines.append(f"{part.upper()} {part}.def")
            extra.append(f"In{part.upper()} in{part}.def")
        elif part == "opttrans":
            for f in ("opttrans.def", "inopttrans.def"):
                shutil.copy(os.path.join(DONOR, f), os.path.join(dest, "inputs", f))
            extra.append("OptTrans opttrans.def")
            extra.append("InOptTrans inopttrans.def")
        elif part == "rbm":
            for section in RBM:
                shutil.copy(os.path.join(DONOR, f"{section}.def"), os.path.join(dest, "inputs", f"{section}.def"))
                lines.append(f"{section} {section}.def")
                src = "inrbmhidden.def" if "Hidden" in section and "PhysHidden" not in section else "inrbm.def"
                extra.append(f"In{section} {src}")
            for f in ("inrbm.def", "inrbmhidden.def"):
                shutil.copy(os.path.join(DONOR, f), os.path.join(dest, "inputs", f))
    lines += ["Orbital orbitalidx.def", "OrbitalParallel orbitalidxpara.def", "TransSym qptransidx.def", ""]
    open(os.path.join(dest, "inputs", "namelist.def"), "w").write("\n".join(lines + extra) + "\n")
    # placeholder fixed file (replaced by the zqp=c_opt variant)
    shutil.copy(os.path.join(ROOT, "tests/fixtures/native_c_physcal_181/heisenberg_chain_real/zqp_opt.dat"),
                os.path.join(dest, "zqp_opt.dat"))
    print(name, "ok")
