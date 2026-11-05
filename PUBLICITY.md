# PUBLICITY: finetune-lab (P10)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set,
each piece graded on the 1,000-point publicity rubric (accuracy 300,
research 200, gap fit 200, clarity 150, reach format 150), ship line
900. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Fan-out set date: 2026-11-05. Project state: v1.0, Learn, Build,
and compressed Deep done 2026-11-05, 39 tests passing. Level
grades on record: Learn 8,200 of 10,000, Build 8,300 of 10,000,
Deep 8,000 of 10,000. Deep detail: a small DPO run on the Build
tabular GRPO policy (beta 0.1, frozen reference, 300 steps) raised
held-out mean margin from -0.0654 to 0.8297 (+0.8951) and held-out
preference accuracy from 0.4375 (7 of 16) to 1.0000 (16 of 16),
with train margin from -0.1123 to 1.6779 and DPO loss from 0.6931
to 0.6076; the forgetting check rose on greedy checker accuracy
from 0.125 to 1.0000 and on fresh-sample mean reward from 0.2070
to 0.5430, with no forgetting observed but named a weak test by
construction; the reward hacking probe found the pattern in
miniature (a rejected-only suppression policy games train
preference accuracy to 1.0000 while held-out stays 0.4375, a
constant action-0 policy scores 0.2500 and cannot game, and the
trained DPO policy is not flagged); and the QLoRA accounting
study (labelled accounting, not measured GPU memory) puts the
stated 7B row at 3,516,777,216 bytes against 14,000,000,000 full
fp16 bytes, fraction 0.2512. All work is toy NumPy teaching work,
not a real LLM: synthetic preferences from a deterministic
checker rule, CPU teaching work on the Muse VM, not BlackBox GPU.

## Channel log (2026-11-05)

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub releases | Live: v0.1 https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v0.1 , v0.5 https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v0.5 , v1.0 https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v1.0 (verified with gh release list, 2026-11-05). Draft release notes for all three are in fanout/release-notes.md. No new release was created for the fan-out. |
| Portfolio index | Held: fan-out links are ready, the index update is a coordinator step outside this cycle. Named reason: held for coordinator refresh. |
| Website page | Draft in fanout/website-page.md. Held: not published to jesunahmadushno.com. |
| Blog | Draft in fanout/blog-post.md. Held: not posted. |
| Tutorial | In repo: fanout/tutorial.md. Every command in it was run and verified on 2026-11-05 (three runners, three per-level test modules, scaffold smoke test, full suite at 39 tests). |
| LinkedIn post | Draft in fanout/linkedin-post.md, marked DRAFT NOT POSTED, link carried as the first-comment line per the LinkedIn rule. Held: not posted, loop never posts. |
| X | Draft held: no X copy is shipped in this set beyond the LinkedIn draft source text. Held under the standing social rules; posting is outside the loop. |
| Facebook | Draft held: no Facebook copy posted. Standing rule applies (no link in body or comments, keyword call to action fulfilled by DM). Held: not posted, loop never posts. |
| Instagram | Draft held: no Instagram copy posted. Standing rule applies (no clickable link in caption, link in bio). Held: not posted, loop never posts. |
| Threads | Draft held: no Threads copy posted. Threads waits for Jesun's tap per post, hash-locked on exact text and media. Held: awaiting Jesun's tap. |
| Video | Held: engine queue, January production for lab projects per the December Compression Protocol. No video was produced in this cycle. |
| Book chapter | Draft angle in fanout/blog-post.md and SPEC.md ("Teaching The Model Manners"). Held: chapter draft follows the finished project set; not a published chapter. |
| Paper candidate | Draft brief in fanout/paper-candidate.md (rejected-only suppression hacking pattern in miniature). Submission is Jesun's tap. Nothing submitted, no venue contacted. The publications tracker is updated by the coordinator, not by this cycle. |

## Publicity grades (1,000-point rubric, ship line 900)

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach format /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| fanout/blog-post.md | 294 | 187 | 194 | 145 | 140 | 960 |
| fanout/tutorial.md | 297 | 184 | 195 | 147 | 142 | 965 |
| fanout/website-page.md | 292 | 180 | 190 | 146 | 144 | 952 |
| fanout/linkedin-post.md | 287 | 174 | 190 | 144 | 140 | 935 |
| fanout/paper-candidate.md | 295 | 196 | 195 | 142 | 130 | 958 |
| fanout/release-notes.md | 295 | 182 | 183 | 144 | 138 | 942 |

All six pieces are at or above the 900 ship line. No piece was
shipped ungraded. Overall fan-out grade: 9,520 of 10,000 (mean of
the six piece totals, 952 of 1,000; the six totals sum to 5,712
and 5,712 divided by 6 is 952).

## Named deductions (unclosable in this cycle)

- Blog post, minus 40: research is condensed from DOSSIER.md v1,
  whose prior art section is brief (minus 13 research); the toy
  synthetic scale limits how far the results travel, stated in
  the piece (minus 10 reach, minus 6 accuracy for reader
  interpretation risk, minus 6 gap fit, minus 5 clarity).
- Tutorial, minus 35: research framing is thinner than the blog
  by design, because the module teaches runs rather than the
  field (minus 16 research); no worked example beyond the shipped
  sets (minus 5 gap fit, minus 3 clarity, minus 8 reach, minus 3
  accuracy).
- Website page, minus 48: condensed research (minus 20),
  condensed gap framing (minus 10), and the page is copy only,
  not a live page (minus 6 reach, minus 8 accuracy, minus 4
  clarity).
- LinkedIn post, minus 65: short form compresses the research
  and the failure mode detail (minus 26 research, minus 13
  accuracy, minus 10 gap fit, minus 6 clarity, minus 10 reach,
  because the post is a held draft and carries no live link in
  the body by rule).
- Paper candidate, minus 42: this is a brief, not a drafted paper
  (minus 20 reach, minus 8 clarity); venue is classed, not named
  (minus 4 research, minus 5 gap fit, minus 5 accuracy for the
  synthetic-only evidence base the paper must carry).
- Release notes, minus 58: notes summarise releases that already
  exist, so research and gap framing are summary grade (minus 18
  research, minus 17 gap fit); format is repo-bound (minus 12
  reach, minus 5 accuracy, minus 6 clarity).

## Jev gate (2026-11-05)

claim_check on the blog main results paragraph, extracted
byte-exact from fanout/blog-post.md by script. Spec:
hidden_files/jev-p10-fanout-claim-spec-2026-11-05.json in the goal
workspace. Results:
hidden_files/jev-p10-fanout-claim-results-2026-11-05.json. Model
jev-1.13.0, question id supported, noul 0.86, route CLEAR (clear
at 0.85 or above, block at 0.10 or below, review between), input
tokens 2168, output tokens 20. Arithmetic was pre-computed in the
spec evidence (0.8297 minus -0.0654 is 0.8951; 7 divided by 16 is
0.4375; 1.6779 minus -0.1123 is 1.7902; 0.6931 minus 0.6076 is
0.0855; 0.5430 minus 0.2070 is 0.3360; 3,500,000,000 plus
16,777,216 is 3,516,777,216 and that total divided by
14,000,000,000 is 0.2512 rounded; 14 plus 14 is 28 and 28 plus 11
is 39). No claim was blocked.
