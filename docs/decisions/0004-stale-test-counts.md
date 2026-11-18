# 0004: The test-count figures in the documents are stale; the suite measures 59

Date: 2026-11-17. Status: accepted.

Context: two totals for the repo test suite stand in the documents,
and both are stale. level-3-deep/README.md says "Tests: 39 total in
the repo suite" and REPORT.md repeats "Tests: 39 in the repo suite"
in its findings; 39 is the three level modules alone (14 Learn, 14
Build, 11 Deep). REPORT.md section 6 (Reproducibility) says the full
suite is "(55 tests, under a second)"; 55 is the level modules plus
the 16 rigor tests, still without the 4 scaffold smoke tests.
PUBLICITY.md also carries the 39 figure. Neither total counts the
whole suite as it now stands.

Decision: the correction of record is the measured count. On
2026-11-17 the documented command, python3 -m unittest discover -s
tests from the repo root (Python 3.12.3, NumPy 1.26.4, the pinned
version), reported "Ran 59 tests" and OK: 59 collected, 59 passed
(14 Learn, 14 Build, 11 Deep, 16 rigor, 4 smoke). The documents
named above are not edited; this entry supersedes their figures.
The preprint already reports the measured count, 59.

Consequences: quote 59 for the repo suite. The 39 and 55 figures
are superseded wherever they appear, including the dated fanout
snapshots that carry the 39.
