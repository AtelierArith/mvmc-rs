"""AST-to-ledger set reconciliation; mappings are not execution evidence."""
import csv
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from inventory_validation import ledger_anchors, validate_rows, validate_sources

root = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--ast', type=Path, required=True)
args = parser.parse_args()
ledger = args.repo / 'docs/reference/c-to-julia/verification'
def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))
ast = read(args.ast)
apis = read(ledger / 'issue-184-public-apis.tsv')
scenarios = read(ledger / 'issue-184-scenarios.tsv')
validate_rows(apis, 'A', 300)
validate_rows(scenarios, 'S', 456)
blobs = {}
reference = args.repo / 'extern/Julia-mVMC'
for package_root, revision, prefix in ((reference,'8bb1b9e8ae47b1512c00b321be05664ddcac0fd1',''),
    (reference/'PfaPack.jl','0dcf52c15caec63516d0703f36bfc8a4bc0e58d0','PfaPack.jl/'),
    (reference/'SFMT.jl','1526553009f318ae78338151460fda78beadddc2','SFMT.jl/')):
    listing = subprocess.check_output(['git','-C',str(package_root),'ls-tree','-r',revision]).decode()
    for entry in listing.splitlines():
        metadata, name = entry.split('\t',1)
        if metadata.startswith('160000 ') or not name.endswith('.jl'):
            continue
        blobs[prefix+name] = subprocess.check_output(['git','-C',str(package_root),'show',revision+':'+name])
validate_sources(ast, blobs)
for row in apis + scenarios:
    ledger_anchors(row, blobs)
def locations(row):
    return {(anchor['source'],anchor['start']) for anchor in ledger_anchors(row,blobs) if anchor['start'] is not None}
exports = []
for item in ast:
    if item['kind'] != 'export':
        continue
    identity = '.'.join(filter(None, (item['module'], item['identity'])))
    exact = [r['id'] for r in apis if r['julia_symbol_or_scenario'] == identity]
    loc = (item['source'], int(item['line']))
    declared_here = [r['id'] for r in apis if loc in locations(r)]
    reexport_identity = 'PfaPack.' + item['identity'] if item['package'] == 'PfaPack' and item['module'] in ('LTLDecomposition', 'Pfaffian', 'Utu2') else None
    reexports = [r['id'] for r in apis if r['julia_symbol_or_scenario'] == reexport_identity] if reexport_identity else []
    bindings = [b for b in ast if b['source'] == item['source'] and b['module'] == item['module']
                and (b['kind'] == 'import_or_alias' or
                     b['kind'] == 'binding' and b['identity'] == item['identity'])]
    exports.append({**item, 'qualified_identity': identity, 'exact_api_rows': exact,
        'declaration_location_rows': declared_here, 'binding_or_import_candidates': bindings,
        'reviewed_reexport_identity': reexport_identity, 'reviewed_reexport_api_rows': reexports,
        'classification': 'DeclaredExportMapped; not overload/runtime proof' if exact else 'ReviewedPfaPackReexportMapped' if reexports else 'UnmappedDeclaredExport',
        'owner': 'Pauli #184 API/source reconciliation', 'execution': 'NotRun'})
tests = []
for item in ast:
    if item['kind'] != 'testset':
        continue
    loc = (item['source'], int(item['line']))
    rows = [r['id'] for r in scenarios if loc in locations(r)]
    tests.append({**item, 'scenario_rows': rows,
        'classification': 'TestsetLocationMapped; original assertions/conditions still require row review' if rows else 'UnmappedTestset',
        'owner': 'Pauli #184 scenario reconciliation', 'execution': 'NotRun'})
includes = [{**r, 'execution': 'NotRun', 'classification': 'IncludeCall; not independently executed scenario'}
            for r in ast if r['kind'] == 'include']
source_files = {r['source']: r for r in ast if r['kind'] == 'file'}
ledger_rows = []
for family, rows in (('public-apis', apis), ('scenarios', scenarios)):
    for row in rows:
        anchors = []
        for source, line in sorted(locations(row)):
            file = source_files.get(source)
            events = [r for r in ast if r['source'] == source and int(r['line']) == line and r['kind'] != 'file']
            anchors.append({'source': source, 'line': line, 'file': file,
                'syntax_events': events, 'classification': 'ExactRevisionSourceBound' if file else 'SourceUnresolved'})
        for source in re.findall(r'(?:^|;\s*)([^;:]+\.jl)', row['julia_source']):
            source = source.strip()
            if not any(a['source'] == source for a in anchors):
                file = source_files.get(source)
                anchors.append({'source': source, 'line': None, 'file': file, 'syntax_events': [],
                    'classification': 'SourceFileBoundWithoutExactLine' if file else 'SourceUnresolved'})
        ledger_rows.append({'family': family, 'id': row['id'], 'kind': row['kind'],
            'validated_line_ranges_and_descriptors': ledger_anchors(row,blobs),
            'identity': row['julia_symbol_or_scenario'], 'anchors': anchors,
            'owner': row['owner'], 'rust_entry': row['rust_entry'], 'settings': row['settings'],
            'command': row['command'], 'existing_result_verbatim': row['result'],
            'reconciliation_result': 'SourceMappingOnly; existing proof must be reviewed separately' if anchors and all(a['file'] for a in anchors) else 'MissingSourceEvidence',
            'authority_or_difference': row['authority_or_intentional_difference'],
            'overload_signatures_source_inventory': [m for m in ast if m['kind']=='method' and any(m['source']==a['source'] for a in anchors)] if family=='public-apis' else [],
            'overload_condition_review': 'UNVERIFIED per-method supported inputs/dispatch/C boundary; signatures are candidate file inventory, not equivalent overloads' if family=='public-apis' else 'NotApplicable; scenario conditions need per-row source/evidence review',
            'proof_owner': row['owner'],
            'new_execution': 'NotRun'})
summary = {'api_rows': len(apis), 'scenario_rows': len(scenarios),
    'ledger_row_classifications': ledger_rows,
    'ledger_rows_source_unresolved': [r['id'] for r in ledger_rows if r['reconciliation_result'] == 'MissingSourceEvidence'],
    'files_by_package': dict(Counter(r['package'] for r in ast if r['kind'] == 'file')),
    'exports_by_package': dict(Counter(r['package'] for r in exports)),
    'testsets_by_package': dict(Counter(r['package'] for r in tests)),
    'declared_exports': len(exports), 'mapped_export_declarations': sum(bool(r['exact_api_rows']) for r in exports),
    'mapped_reexport_declarations': sum(bool(r['reviewed_reexport_api_rows']) for r in exports),
    'unmapped_export_declarations': sum(not r['exact_api_rows'] and not r['reviewed_reexport_api_rows'] for r in exports),
    'testset_occurrences': len(tests), 'mapped_testset_locations': sum(bool(r['scenario_rows']) for r in tests),
    'unmapped_testset_locations': sum(not r['scenario_rows'] for r in tests),
    'include_calls': len(includes),
    'limits': ['Static AST declarations include conditional/nested exports; runtime module availability is not evaluated',
        'Aliases/import candidates require source review; name matching does not establish equivalent APIs',
        'Testset location match is not semantic assertion/loop/execution coverage',
        'Examples and dynamically composed entrypoints still require source-level scenario reconciliation',
        'No existing ledger classifications promoted and no numerical runs'],
    'exports': exports, 'testsets': tests, 'include_calls_inventory': includes}
print(json.dumps(summary, indent=2, sort_keys=True))
