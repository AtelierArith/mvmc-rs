import copy
import hashlib
import unittest
from inventory_validation import REVISIONS, ledger_anchors, validate_rows, validate_sources

class InventoryControls(unittest.TestCase):
    def setUp(self):
        self.blobs = {'src/A.jl': b'module A\nexport f\nend\n'}
        self.sources = [dict(source='src/A.jl', kind='file', line='1', sha256=hashlib.sha256(self.blobs['src/A.jl']).hexdigest()),
                        dict(source='src/A.jl', kind='export', line='2', sha256=hashlib.sha256(self.blobs['src/A.jl']).hexdigest())]
        for row in self.sources:
            row.update(package='Julia-mVMC',revision=REVISIONS['Julia-mVMC'])
        self.row = dict(id='A001', julia_source='src/A.jl:2', julia_symbol_or_scenario='A.f',
            owner='Pauli', rust_entry='Unmapped', settings='NotRun', command='source review',
            result='UNVERIFIED', authority_or_intentional_difference='C review pending')
    def test_source_bound_is_not_semantic_pass(self):
        self.assertEqual(validate_sources(self.sources, self.blobs), 'SourceBoundOnly')
        self.assertIn('not runtime', validate_rows([self.row], 'A', 1))
    def test_corrupt_source_hash_rejected(self):
        rows=copy.deepcopy(self.sources); rows[1]['sha256']='0'*64
        with self.assertRaises(ValueError): validate_sources(rows,self.blobs)
    def test_changed_source_rejected(self):
        with self.assertRaises(ValueError): validate_sources(self.sources,{'src/A.jl':b'changed\n'})
    def test_invalid_line_types_ranges_rejected(self):
        for line in ('0','4','2.0','true','-1',2,True):
            with self.subTest(line=line):
                rows=copy.deepcopy(self.sources); rows[1]['line']=line
                with self.assertRaises(ValueError): validate_sources(rows,self.blobs)
    def test_missing_and_duplicate_source_rejected(self):
        for rows in (self.sources[1:],self.sources+[self.sources[0]]):
            with self.assertRaises(ValueError): validate_sources(rows,self.blobs)
    def test_unknown_source_rejected(self):
        rows=copy.deepcopy(self.sources); rows[1]['source']='elsewhere.jl'
        with self.assertRaises(ValueError): validate_sources(rows,self.blobs)
    def test_unknown_record_kind_rejected(self):
        rows=copy.deepcopy(self.sources); rows[1]['kind']='VERIFIED_PASS'
        with self.assertRaises(ValueError): validate_sources(rows,self.blobs)
    def test_missing_extra_duplicate_ledger_rejected(self):
        for rows in ([],[self.row,self.row]):
            with self.assertRaises(ValueError): validate_rows(rows,'A',1)
        with self.assertRaises(ValueError): validate_rows([self.row,self.row],'A',2)
    def test_wrong_ledger_identity_rejected(self):
        row=dict(self.row,id='A002')
        with self.assertRaises(ValueError): validate_rows([row],'A',1)
    def test_missing_owner_or_proof_field_rejected(self):
        for field in ('owner','result','command','settings'):
            row=dict(self.row); row[field]=' '
            with self.assertRaises(ValueError): validate_rows([row],'A',1)
    def test_wrong_revision_and_package_rejected(self):
        for key,value in (('revision','0'*40),('package','PfaPack'),('revision',None)):
            rows=copy.deepcopy(self.sources); rows[1][key]=value
            with self.assertRaises(ValueError): validate_sources(rows,self.blobs)
    def test_ledger_outside_or_malformed_numeric_anchor_rejected(self):
        for suffix in ('999','0','2.0','-1','true','1-999','3-2','1,999',''):
            row=dict(self.row,julia_source='src/A.jl:'+suffix)
            with self.assertRaises(ValueError): ledger_anchors(row,self.blobs)
    def test_ledger_range_list_and_file_only_explicit(self):
        self.assertEqual(len(ledger_anchors(dict(self.row,julia_source='src/A.jl:1-2,3'),self.blobs)),2)
        self.assertEqual(ledger_anchors(dict(self.row,julia_source='src/A.jl'),self.blobs)[0]['classification'],'FileOnly')
    def test_unknown_file_and_descriptor_rejected(self):
        for source in ('missing.jl:1','src/A.jl:anywhere'):
            with self.assertRaises(ValueError): ledger_anchors(dict(self.row,julia_source=source),self.blobs)

if __name__ == '__main__': unittest.main()
