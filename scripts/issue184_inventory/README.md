# Issue184 source-set reconciliation

Related to #184 and #185. Metadata-only developer tools: no package code is
evaluated, no model/reference execution or fixture regeneration occurs, and
Cargo does not invoke these scripts. This is an inventory/proof-gap artifact,
not completion of the numerical or umbrella acceptance criteria.

## Source identities and reproduction

The parser reads exact Git objects, not dirty reference files:

* Julia-mVMC `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`;
* nested PfaPack `0dcf52c15caec63516d0703f36bfc8a4bc0e58d0`;
* nested SFMT `1526553009f318ae78338151460fda78beadddc2`.

All three commits must exist in the respective reference repositories. Julia
1.13.1 is used only for `Meta.parseall`; imported stdlib SHA is not a model
oracle. Python uses uv with no project/dependency installation. Run from the
Rust repository, using a new exclusive output directory:

```sh
inventory_output=$(mktemp -d /tmp/mvmc-184-inventory.XXXXXX)
julia +1.13.1 --startup-file=no scripts/issue184_inventory/ast_inventory.jl \
  "$PWD/extern/Julia-mVMC" > "$inventory_output/ast.tsv"
uv run --no-project --no-python-downloads python -B \
  scripts/issue184_inventory/reconcile_sets.py --repo "$PWD" \
  --ast "$inventory_output/ast.tsv" > "$inventory_output/reconciliation.json"
uv run --no-project --no-python-downloads python -B \
  scripts/issue184_inventory/focused_anchors.py --repo "$PWD" \
  > "$inventory_output/focused12.json"
uv run --no-project --no-python-downloads python -B \
  scripts/issue184_inventory/test_inventory_validation.py -v
julia +1.13.1 --startup-file=no scripts/issue184_inventory/ast_inventory.jl --self-test
sha256sum scripts/issue184_inventory/* "$inventory_output"/*
```

Keep commands, tool versions, terminal codes, script and ledger input hashes
with each generated artifact. Changed ledger classifications change output;
historical generated output must retain its original ledger snapshot. The
tools print output only; they never rewrite shared ledger TSVs.

## Actual bounded evidence

The repaired-source receipt is `/tmp/mvmc-184-inventory-final.rgD16e`,
handle50994, aggregate0:14 Python controls and5 Julia AST type controls passed;
AST generation, strict source/ledger reconciliation and focused anchors each0.
Source and input pre/post were identical. Actual commands, versions, logs,
individual terminals and artifact SHA are retained there. This is metadata-only.
The immediately preceding cX6Lg4 receipt failed because conditional `using Test`
did not make its macro available during lowering; that failed receipt is retained.
The repaired import is top-level. No model was run in either attempt.

The three ledger inputs were independently compared to committed main
`50b7318057704130fd4e2901075b0bc9ca1352f9` and are byte-identical:

| Committed input | SHA256 |
| --- | --- |
| issue-184-public-apis.tsv | 1351df60decc0be0502ef21b55cd95a6d2ec8e906ece468674305564f99e5276 |
| issue-184-scenarios.tsv | a1f4eca6e3504fddc9fa2fe1f6cd14016d005d2a14cac9b90a036cd7d80958b1 |
| issue-184-sources.tsv | a81a31f5eeae68d5fc737347a2336f162768f312998e9f67fc6d3295e985277a |

These hashes—not the separately dirty CURRENT prose—identify the inventory's
ledger baseline. Earlier receipts below retain their original source/limitations;
they are not validation of the repaired type/package/revision/anchor checks.

Initial external packet `/tmp/mvmc-184-current-audit.BPDES2` used Julia1.13.1
and uv0.12.21/Python3.12. The method-aware AST generation, source-bound set
reconciliation and focused twelve-anchor generation completed terminal0;
10 synthetic validation controls passed in0.001s. This is not a Rust test run.

Generated artifact SHA256:

| Artifact | SHA256 |
| --- | --- |
| ast_inventory-methods.tsv | 8e0781ac8614166a275bdeda388e91368d3054a15e93ba6e7aa60bf86075ca51 |
| method-reconciliation.json | 7aa874fa43f7d52ef7368ebe4c772cbeb12b450fc9a4b7d8ae3872c60129250c |
| focused12.json | 310c7469a500853e9cad339417a6ae24090d00492ebeccb8c831c3bbaf220e74 |

An initial file scan failed on nested gitlinks; those were subsequently
explicitly inventoried at their exact revisions. The failure is not a model
failure. An intermediate file-path parser incorrectly stopped at a directory
name ending `.jl`; it was corrected before the reported source-set artifact.

## Classification, ownership and remaining work

The validator binds every AST row's package/revision to its actual parent or
nested source namespace and blob SHA; wrong labels fail. Numeric ledger anchors
include comma lists and ranges: every endpoint must be a positive integer within
the exact file. File-only anchors are explicit; the three historical human
descriptors S454–S456 are narrowly whitelisted as descriptor-only, not numeric
proof. Other malformed descriptors/unknown source/invalid lines fail. `struct`
identity uses args[2], not its mutable Bool; abstract/primitive names use args[1].
Controls cover plain/mutable/abstract/primitive declarations, malformed type,
wrong package/revision, line999/reversed ranges/float/bool spellings, duplicate/
missing identities and source hashes. These checks establish source binding only.

125 Julia files parsed:109 parent,10 PfaPack,6 SFMT. 51 declared export
occurrences map to45 qualified API declarations plus6 reviewed PfaPack
submodule reexports through the root's explicit `using` statements. SFMT's
renamed C dump import is not the public dump wrapper. Imports, bindings and
method signatures are retained, not assumed to establish dispatch equivalence.

All313 original testset source locations map to the313 existing testset rows.
The full ledger has300 API and456 scenario rows; these also include constants,
types, constructors, script entries, expanded model/MPI cases and Rust safety
extensions. All756 rows are source-file-bound; twelve original file-only rows
have focused proposed numeric anchors. Source location/name matching is not
assertion, overload, supported-input or runtime verification.

`focused12.json` retains original results/settings/commands/owners separately
from proposed anchors and explicit proof gaps. In particular S452–S456 must
not claim original Julia tests cover Rust asymmetric-error/control consensus
contracts. S446 records the original data-only writer mismatch separately from
Wegener's prospective writer repair. No historical PASS is reassigned to6af.

Pauli owns nested/export/API/scenario reconciliation and overload/condition
gaps. Chandra owns the seventeen documentation/example source mappings and
shared TSV merge. Unreviewed per-method conditions remain UNVERIFIED; no row
is promoted by this tool. Existing1,444 assertion categories are not executed
case counts. Dynamic export/metaprogram availability, method dispatch/failure
boundaries, expanded example inputs, MPI worker/group matrices and per-row
independent evidence still require review. InterAll exclusions remain scoped
to their actual blocks, not blanket file exclusions. No performance claim.
