# PUBLICITY: finetune-lab (P10)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Fan-out set date: 2026-11-05. Project state: v1.0, Learn, Build,
and compressed Deep done 2026-11-05, 39 tests passing. Deep detail: a small DPO run on the Build
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

## Fan-out set (moved from README.md)

Written fan-out set for v1.0, dated 2026-11-05, in fanout/:

- fanout/blog-post.md: "Teaching The Model Manners", the full
  story with the main results claim gated by Jev claim_check
  (model jev-1.13.0, noul 0.86, route CLEAR, no block).
- fanout/tutorial.md: course module P10-finetune-lab. Every
  command in it was executed and passed on 2026-11-05 (all three
  runners, all per-level tests, full suite at 39 tests).
- fanout/website-page.md: project page copy only, not published.
- fanout/linkedin-post.md: DRAFT NOT POSTED, link as the
  first-comment line per the LinkedIn rule.
- fanout/paper-candidate.md: rejected-only suppression hacking
  pattern note at CANDIDATE. Submission is Jesun's tap, nothing
  submitted.
- fanout/release-notes.md: draft notes for the live releases
  v0.1, v0.5, and v1.0.

 The loop never posts to social:
all social channels are held with named reasons in PUBLICITY.md.

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
| Book chapter | Draft angle in fanout/blog-post.md and PROPOSAL.md (original scope) ("Teaching The Model Manners"). Held: chapter draft follows the finished project set; not a published chapter. |
| Paper candidate | Draft brief in fanout/paper-candidate.md (rejected-only suppression hacking pattern in miniature). Submission is Jesun's tap. Nothing submitted, no venue contacted. The publications tracker is updated by the coordinator, not by this cycle. |

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
