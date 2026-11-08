# 0002: Code stays in the stage folders

Date: 2026-11-08. Status: accepted.

Context: the portfolio standard v2 puts code in src/ and evidence in
experiments/ and results/, but allows domain-named folders listed in
the README layout table.

Decision: keep level-1-learn/, level-2-build/, level-3-deep/ and rigor/
as they are. Each holds its runner, its method note and the results
JSON the runner writes.

Consequences: no import paths, test paths or dashboard imports change,
and the tags v0.1 to v1.0 still point at the same layout. The README
layout table maps each folder to its stage.
