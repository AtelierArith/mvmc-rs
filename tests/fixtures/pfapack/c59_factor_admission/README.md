# Independent C59 inputs for a diagnostic-only CI acquisition

These four JSON files are byte copies of the separately approved SOURCE128
data-only extraction. They include retained original input, native factor,
inverse, negative-inverse planes and provenance/qualification; none is generated
from Rust results. `input_nN.words` is a lossless selection of `inputWords`
from the corresponding independently hashed JSON, with a fixed header.
The Node loader hashes each full JSON BEFORE parsing and verifies the derived
word-file bytes against that input array before starting the actual producer.
The Rust ignored driver embeds only those word files. No oracle/runtime
dependency is introduced into Cargo and no existing golden file is modified.

Provenance remains historical C59 GNU 13.3 / LP64 integer32 standalone
DSKTF2 U/N and original inverse-helper acquisition, not current full C
sampling/SR/PhysCal parity. Compile-r4 terminal1 (parser EPIPE), successful
r5 parser qualification, n32 r6 terminal1 (schema-reader bug), r7 read-only
qualification, and n64/128/256 r8 successful acquisitions remain distinct.
See unchanged PROVENANCE.json, extraction-inputs.json and each JSON's
qualification/provenanceFiles for the retained original bindings/notices.

This change uses the data as independently sourced INPUT only. It does not
adopt their computed output bits as ordinary-test expectations, migrate the
historical Julia assertions, choose tolerances, or waive the four real fixture
failures / eight workspace callback failures. SOURCE120 condition fractions
are independent reference uncertainty evidence, not an automatic candidate
comparison threshold. Factor/Pf/inverse scientific acceptance remains pending
the source-associated factor/inverse DAG and independent conditioning proof.

The first acquisition is only Linux default/scalar, profile ci. SIMD, BLAS,
both-feature and original FSZ/sampling/CG paths require their own proofs.
