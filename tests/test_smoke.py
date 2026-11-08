"""Layout test: the files the portfolio standard requires exist, and no public file carries a grade.

The repo is not pre-registered (docs/decisions/0003), so there is no lock to pin.
When a confirmatory PROPOSAL.md is locked, add its SHA-256 pin here.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ["README.md", "PROPOSAL.md", "REPORT.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
            "requirements.txt", "scripts/check_results.py",
            "level-1-learn/results_learn.json", "level-2-build/results_build.json",
            "level-3-deep/results_deep.json", "rigor/results_rigor.json",
            "docs/decisions/0001-keep-recorded-grpo-results.md"]
# "x of 10,000", "x/10000", "Grade:", rubric scores like "7/10" or "/1000".
GRADE = re.compile(r"\d[\d,]* of 10,?000|/\s?10,?000\b|\bGrade:|\bgrade \d|\b\d+/10\b|/1000\b|minus \d[\d,]* points", re.I)


def public_texts():
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in {".git", "node_modules", "dist", ".rubric", "__pycache__", ".pytest_cache"}]
        for f in files:
            if f.endswith((".md", ".jsx", ".js", ".html", ".cff")):
                yield os.path.join(d, f)


class Layout(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ROOT, p))]
        self.assertFalse(missing, "missing: " + ", ".join(missing))

    def test_readme_opens_with_the_abstract(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            text = fh.read()
        for field in ("**Question.**", "**Headline.**", "**Field.**", "**Job family.**", "**Verdict.**", "## Lifecycle", "## Layout"):
            self.assertIn(field, text)

    def test_no_public_grade(self):
        hits = []
        for p in public_texts():
            with open(p, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if GRADE.search(line):
                        hits.append(f"{os.path.relpath(p, ROOT)}:{n}: {line.strip()[:80]}")
        self.assertFalse(hits, "grades are internal only:\n" + "\n".join(hits))

    def test_grade_pattern_catches_a_grade(self):
        for bad in ("Grade: 8,000 of 10,000", "8200/10000", "| Method | 7/10 |"):
            self.assertTrue(GRADE.search(bad), bad)


if __name__ == "__main__":
    unittest.main()
