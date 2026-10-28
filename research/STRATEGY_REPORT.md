# More Than A Hack 2027: research, technical assessment and execution plan

Assessment date: **21 October 2026**, Europe/London. This is a research report, not a trained solution or a prediction of qualification. No competition solver has been implemented. Implementation advances only when you explicitly tell me to continue.

**28 October update:** the validation advice and calendar in this historical assessment are superseded by [the revised implementation plan](<D:/Ai agents Projects/More  then  HACK/solution/docs/VALIDATION_UPDATE_2026-10-28.md>). In particular, do not implement the supplied-match annotation proposals below. Current online rules differ from the downloaded snapshot used for this assessment.

## 1. Recommendation and evidence status

Build an **offline, uncertainty-aware possession segmentation system** using tactical tracking, Smart Tagging and explicit alive/dead evidence. Begin with a small state model and add learned components only when independent validation supports them. Preserve a deterministic state-machine fallback.

The strongest immediate opportunities are: distinguish a missing ball from a dead ball; reject brief opponent touches as possession changes; combine uncertain team evidence across time; recover stoppage and restart boundaries. This choice follows the supplied data and scorer, rather than a preference for sophisticated models.

Evidence labels used throughout:

- **Verified locally:** inspected files or executed measurements.
- **Official:** organizer documentation or event/partner pages.
- **Research evidence:** results on other datasets, not a demonstrated challenge score.
- **Hypothesis:** proposed improvement requiring experiments.

There is no honest numerical estimate of our eventual rank. The competitors' methods and the full-match answers are unknown. Beating the published baseline is necessary progress, not evidence that top 20 is secured.

## 2. Competition intelligence

### Organization, purpose and progression

Mobile World Capital Barcelona, Barça Innovation Hub and Metrica Sports organize this football analytics competition. Talent Arena is the digital talent event hosting the final, within the wider MWC Barcelona ecosystem. This is an applied sports-data challenge, not a requirement to use telecom APIs. The broader institutional aim is to connect technical talent with practical innovation. [Official competition page](https://landing.mobileworldcapital.com/more-than-a-hack-2027), [MWCapital overview](https://mobileworldcapital.com/en/about-us/).

MWCapital is a public/private foundation whose trustees include public administrations, Fira de Barcelona, GSMA and corporate partners. That does **not** establish who pays each hackathon expense. I found no itemized 2027 competition funding agreement; the funding split is unverified. [Foundation information](https://mobileworldcapital.com/en/about-us/).

| Stage | Published dates | Progression |
|---|---|---|
| Registration | 1 October–10 November 2026 | Teams of 3–5; all use the same registered name |
| Initial Challenge | 20 October–17 November 2026 | Automatic ranking; top 20 advance |
| Innovation Challenge | 26 November 2026–14 January 2027 | Broader analytical/tactical work; top five selected |
| Final pitch | 2 March 2027 | In person at Talent Arena, Barcelona |

The later stage offers ten professional matches and technical mentoring. The five finalists receive covered Barcelona travel/accommodation and listed Barça benefits. Limits on reimbursements, visas, booking procedure and the detailed later-stage rubric were not verified. No cash award is confirmed here. [Competition stages and benefits](https://landing.mobileworldcapital.com/more-than-a-hack-2027).

**Correction to the initial expectation:** top 20 qualifies for the online Innovation Challenge; top five from that stage are invited to Barcelona. A team with fewer than three members is ineligible for the later stages. [Partner announcement](https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data).

Your email supplies the exact deadline, **17 November 2026, 23:59 CET**, equivalent to **22:59 London time** that day. The repository confirms the date, but not that exact clock time. Use the earlier internal deadline below and recheck the submission email. Submission instructions are expected by 14 November according to your email; the repository already describes the three deliverables, but the actual form link remains email-only.

### History and what it actually tells us

Only **one preceding edition, 2026**, was reliably documented. The 2026 official starter calls it the Soccer GPT Hackathon and supports tactical analytics and Nexus playlist exports. Its format was open-ended; the 2027 overview identifies originality, technical solidity, LLM use, football relevance and presentation as historical criteria. None is a separate scoring component in the 2027 qualifier. [2026 official starter](https://github.com/metrica-sports/more-than-a-hack), [2027 official overview](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md).

| Historical result | What is supported | Limit |
|---|---|---|
| IpurdIAr / IpurdIArs, winner | A team member's first-person account and embedded BIHUB announcement describe a chatbot connecting match tracking and commentary to tactical insights. | Exact model, training procedure, source code and judge score breakdown not verified. |
| Pol Pujol Santaella and teammates, third place | First-person account describes tactical metrics, generated narratives, a vector knowledge base and tactical RAG. | Self-reported placement; not a complete official results table. |

Sources: [winner account with BIHUB announcement](https://www.linkedin.com/posts/guillemolivart_ipurdiar-we-have-created-a-chatbot-that-activity-7457345043708665856-PeSQ), [third-place account](https://www.linkedin.com/posts/polpujolsantaella_generativeai-sportsanalytics-footballanalytics-activity-7435001650068656129-BhXz).

I did not verify the runner-up, all five finalists, numerical judging weights, or detailed judges' feedback. Applicant totals reported for Talent Arena and for this challenge are not interchangeable, so I do not use them to estimate our odds. One edition is insufficient evidence of enduring jury preferences. The useful historical lesson is that understandable tactical answers have value; copying a chatbot does not help this year's automatic qualifier.

### Metrica's relevance

Metrica describes Nexus for professional clubs and PlayBase for coaches/academies. Its public materials connect video, automatic player/field tracking, tactical patterns and coach-facing workflows. Public sample data and the Friends of Tracking resources provide useful development examples. [Metrica's 2027 announcement](https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data), [data-driven video analysis](https://www.metrica-sports.com/blog/data-driven-video-analysis), [sample-data repository](https://github.com/metrica-sports/sample-data).

The competition's own overview says possession segmentation is a production problem for Metrica. We can reasonably infer that reliable boundaries and graceful handling of noisy data are useful. That inference is not a hidden judging criterion. I did not verify the proprietary production model architecture and will not invent it.

## 3. Repository and dataset audit

### What I inspected

The downloaded repository is a source snapshot, **not a Git checkout**. I inspected every supplied documentation file, all Python scripts, the source cells of all three notebooks, the public labels/metadata, baseline CSVs through the scorer, requirements and ignore rules. No AGENTS.md was found in the project inventory.

Key local specification files:

- [Challenge definition](<D:/Ai agents Projects/More  then  HACK/more-than-a-hack-2027-main repo/CHALLENGE.md>)
- [Data and alignment](<D:/Ai agents Projects/More  then  HACK/more-than-a-hack-2027-main repo/DATA.md>)
- [Evaluation explanation](<D:/Ai agents Projects/More  then  HACK/more-than-a-hack-2027-main repo/EVALUATION.md>)
- [Executable metric specification](<D:/Ai agents Projects/More  then  HACK/more-than-a-hack-2027-main repo/evaluation/possession_metrics.py>)
- [Rules](<D:/Ai agents Projects/More  then  HACK/more-than-a-hack-2027-main repo/RULES.md>) and [submission contract](<D:/Ai agents Projects/More  then  HACK/more-than-a-hack-2027-main repo/SUBMISSION.md>)

The repository has a nearest-player baseline, loaders for tracking/tags/homography, a CSV validator, the actual ranking script and three instructional notebooks. It has no provided trained possession model or full-match public answers.

I streamed both tracking files and parsed both metadata files, both homography tables and the tagging XML directly from eight ZIP archives. The archive README was read locally without sending its contents to an LLM. Both video containers were inspected for metadata. **I have not decoded or watched the full videos.** Dataset record inspection used local computation; no trajectories, clips or restricted raw records were sent to web services.

### Locally verified assets

| Asset/property | Tactical | Broadcast |
|---|---:|---:|
| Tracking records | 199,319 | 164,488 |
| Raw frame IDs | Exactly 1 through record count | Exactly 1 through record count |
| Track slots in metadata | 46 | 51 |
| Ball present, all frames | 113,488 / 56.94% | 83,121 / 50.53% |
| Homography rows | 199,319 | 164,488 |
| Valid homographies | 100% | 65.81% |
| Local video size | 5,222,307,334 bytes | 13,962,360,320 bytes |
| Container video samples | 199,319 | 164,487 |

Tactical container metadata reports 1920×1080 and 7,972.76 seconds, consistent with 25 fps and the required frame count. The broadcast container reports **one fewer video sample than its tracking rows**. This is a metadata-level discrepancy, not yet a decoded-frame diagnosis. Do not pad a stream to conceal it; verify with an actual decoder before using broadcast fusion. Tactical-only development is unaffected.

The local video names have download suffixes: `BAR-ATH_tactical-002.mp4` and `BAR-ATH_broadcast-001.mp4`. The starter expects canonical names within tactical/broadcast directories. The other assets are still archived. The files are present, but the default loader paths are not ready. These ZIPs each contain separate files; they are not binary pieces to concatenate into one archive.

The tactical stream has **158,857 graded frames**; the ball is present on **94,214**, or about **59.31%** of those. Mean player observations during graded time: **20.49**. There are 150 contiguous missing-ball stretches within the graded blocks: median **10.72 s**, 95th percentile **50.80 s**, maximum **221.48 s**. Those include dead-ball periods and the graded pre-kickoff interval: they are **not** statistics of live-play occlusion alone. Nevertheless, short-gap interpolation cannot address the whole problem.

Smart Tagging has **638 intervals**, including **59 set pieces**. Its Team field contains Barcelona 314 times, `Atletic Club` 288 times and N/A 36 times. All throw-ins lack a team label according to the supplied documentation. Defensive tags name the defending team: naively assigning that team possession would reverse their meaning.

### Target, definitions and alignment

This is one match, **Barcelona home versus Athletic away**, LaLiga 2024–25 matchday 2, with two views of that same match. It is not a collection of independent matches or two independent evaluation examples.

The required CSV is exactly:

```text
frame,label
0,X
1,X
...
199318,X
```

Exactly **199,319 data rows**, frame IDs **0–199318** in order, labels **H/A/D/X**, no blank labels, no extra index column. Use the exact canonical header even though the parser tolerates case/whitespace. Successful validation checks file structure, not prediction quality.

- H: home possession while alive.
- A: away possession while alive.
- D: ball dead; restart ownership is not requested.
- X: abstention or ungraded time, with different consequences under the two metrics.

While play is alive there is always an owning team. Loose/contested balls retain the last controlled owner; deflections and uncontrolled touches need not change it. A stoppage ends the current possession even if the same team restarts. The reference labels are mechanically derived from **corrected tracking's ball state and ownership**, without subsequent smoothing. We must approximate that convention, which may differ from a viewer's interpretation.

| Input | Submission alignment |
|---|---|
| Tactical ATD `frame_id` | Subtract **1** |
| Tactical tagging seconds | `round(seconds * 25)` |
| Tactical homography `frame` | Already zero-based |
| Broadcast | Separate timeline; no mapping supplied |

Do not apply the million-frame per-half offsets in games.py to the delivered automatic tracking. Those describe the separate corrected-tracking counter used by the organizers.

Graded intervals, inclusive: **11185–90651** and **110808–190197**. Public answers cover only **11185–18684**: H 3,454 frames, A 2,577, D 1,469; **17 possessions, 21 segments**. The second graded block begins before actual second-half kickoff; **110808–113741 is graded D**, and kickoff is 113742. This is explicitly published, so it is legitimate boundary information. Outside graded blocks, output X.

### Exact metric behavior

The code ranks by **spell_f1_mean**, displayed as spellF1. Within each contiguous GT-labelled block it fills predicted X from the preceding non-X, or from the first known label at the block's head. It then extracts maximal H/A intervals. D intervals are not possession objects but split adjacent possessions.

Same-team predicted and true spells receive temporal intersection-over-union:

`IoU = overlapping duration / union duration`.

At each threshold 0.3, 0.4, 0.5, 0.6 and 0.7, the scorer greedily matches descending-IoU pairs one-to-one. F1 is `2 TP / (2 TP + FP + FN)`; spellF1 is the mean of these five values. This is not confidence-ranked average precision, a boundary tolerance score, or match possession percentage.

Tie-breaks in rank_key: higher macroF1, then higher transition F1 at the loose tolerance (default 25 frames), then smaller mean absolute transition latency, then filename. macroF1 averages H/A/D F1 and treats graded X as a miss. Some docstrings still incorrectly call macroF1 the ranking metric; the actual rank_key and RANK_METRIC settle this.

Consequences: brief flicker creates extra predicted possessions; excessive smoothing merges real ones; incorrect D splits or joins spells; boundary accuracy matters most on short spells. X is not a safe escape at transitions. We should normally emit the inferred state on graded frames, improving the tie-break without introducing unnecessary abstentions.

### Executed baseline check

Ran the unchanged supplied scorer with `--bootstrap 1000` using Python **3.11.9**, NumPy **2.4.3**, pandas **3.0.1**. All four CSVs validated and reproduced the published public scores:

| Baseline | Public macroF1 | Public spellF1 | Public predicted spells | Official reported full-match spellF1 |
|---|---:|---:|---:|---:|
| Raw nearest player | 0.5972 | 0.0378 | 131 | 0.0486 |
| Smooth ±25 frames | 0.6200 | 0.2868 | 36 | 0.1996 |
| Smooth ±50 frames | 0.6207 | 0.3256 | 26 | **0.2433** |
| Smooth ±125 frames | 0.5812 | **0.4242** | 16 | 0.2242 |

I **did not run private evaluation or regenerate the baseline predictions from tracking**. Full-match values above are organizer-reported. The smoothing parameter is a half-width; ±50 means a 101-frame window, not a 50-frame window. The implementation counts H/A votes, excluding missing observations; it emits D only when no H/A votes remain. It is not a three-class majority vote.

The official bootstrap's 5th–95th percentile public spellF1 intervals are **0.2140–0.4465** for smooth50 and **0.2667–0.5939** for smooth125. Their overlap and the documented public/private reversal warn against choosing on this sample. These are conditional estimates on a tiny sample, not confidence in full-match generalization; the repository's approximate 0.04 full-match noise band is not a universal significance threshold.

### Rules affecting our design

Pretrained models, external data, classical models and LLMs are permitted if inspectable, accessible and explained. Proprietary unavailable datasets/models fail that requirement. The repository must regenerate the **same submission file**, not merely similar scores.

Hand-editing submitted predictions is forbidden. Small independently annotated training/validation samples are expressly permitted by RULES.md and OVERVIEW.md, despite FAQ shorthand saying no to hand-labelling. We will keep human reference labels separate, never copy them into the final output and never use other corrected possession data for this fixture.

The match assets are separately licensed for this competition. No public redistribution of video, tracking, reconstructive derivatives or clips. The source repository's MIT licence does not grant dataset rights. FAQ/OVERVIEW additionally prohibit sending restricted assets or extracts to hosted/cloud LLMs. Public documents and code can support our coding discussions; match processing remains local. Signed registration terms are not present locally and still need your review for any additional conditions.

There is **no live leaderboard and no submission feedback**. One editable team form response requires the team CSV, public code URL and short PDF/Markdown technical document. The team name should appear in the CSV/document filename. This is confirmed in the local submission contract.

## 4. Applicable research, with practical limits

No cited paper has been evaluated on this challenge's hidden corrected labels. Their relevance is methodological; paper benchmark scores cannot be translated into expected challenge scores.

| Method and source | Useful idea | Fit, cost and limitation |
|---|---|---|
| [Vidal-Codina et al., automatic event detection, 2022](https://arxiv.org/abs/2202.00804) | Possession inference followed by football-rule event detection; set-piece configurations contribute context. | Strong reason to establish a transparent geometry/rules baseline. CPU practical; moderate engineering. Its event detection results do not establish possession spellF1 under missing automatic tracking. |
| [MS-TCN, CVPR 2019](https://arxiv.org/abs/1903.01945) | Dilated temporal convolutions and smoothing loss reduce over-segmentation. | Small feature-based TCN is feasible on the RTX 3050; data sufficiency is the issue. Low/medium compute, medium implementation effort. Full video feature extraction would be a separate expensive task. |
| [ASFormer, BMVC 2021](https://arxiv.org/abs/2110.08568) | Temporal attention with local/hierarchical structure refines action segments. | Sensible later comparator, but 17 labelled possessions do not justify a large model. Higher tuning and memory risk; published action datasets differ from football. |
| [Ball Radar, KDD 2023](https://arxiv.org/abs/2306.08206), [author code](https://github.com/hyunsungkim-ds/ballradar) | Player-set representations plus bidirectional sequence models infer possessor and ball trajectory. | Directly relevant to missing balls. Source uses older PyTorch; some original training data is private, although authors provide a public Metrica route. Use only a fully accessible reproduction path. Slot identity fragmentation requires adaptation. Medium/high effort. |
| [PathCRF, KDD 2026](https://arxiv.org/abs/2602.12080), [author code](https://github.com/hyunsungkim-ds/pathcrf) | A dynamic player graph and constrained CRF/Viterbi inference make ball-free possession paths logically consistent. | Strongest experimental direction. Repository supplies trained model variants and an open seven-match Sportec route. Reproduction, checkpoint terms and transfer must be checked. Complete stable-player trajectories differ substantially from these fragmented slots. Medium GPU cost, high adaptation risk. |
| [Multi-Modal Soccer Scene Analysis, WACV 2026](https://arxiv.org/abs/2512.19528) | Combine trajectories, player types and image crops through temporal/spatial transformers and masked pretraining. | Relevant to trajectory/state/possessor inference; image decoding/cropping and supervision increase cost. A research reference, not a ready dependency; checkpoint/data access has not been established. |
| [Renner et al., October 2026 preprint](https://arxiv.org/abs/2610.11571) | GNN plus sequential modeling captures tactical configurations. | Relevant to later tactical analysis. It uses possession and ball-status indicators as inputs, so it cannot be cited as solving our missing possession target. Preprint evidence on a different task. |

The Metrica [sample-data repository](https://github.com/metrica-sports/sample-data) offers synchronized event/tracking examples and links to [Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking). These can support development and external-match sanity checks after checking their own licences. Their events are not automatically the same H/A/D frame ground truth; conversions require documented assumptions and timing validation.

All of these algorithm families are permitted in principle by the challenge. A specific dependency's code/weight/data licence and organizer accessibility still need verification before adoption. Do not use an otherwise promising checkpoint if its reproduction relies on inaccessible private assets.

## 5. Ranked strategy comparison and selected architecture

The ranking below is **expected value under our time, hardware and supervision constraints**, not an experimentally measured score ranking.

| Rank / candidate | Expected metric effect and evidence | Reliability / originality | Cost, complexity, feasibility | Competition and later-stage fit |
|---|---|---|---|---|
| **1. Tracking + tags + structured temporal decoder** | Hypothesis: fewer false switches and better stoppage boundaries. Supported by baseline failure modes and structured-inference research. | Explicit missingness and weak-source reliability can generalize; rule calibration remains risky. Useful engineering originality, no novelty claim. | CPU; medium effort; achievable by early November. | Directly targets spells; reusable interval/confidence foundation. |
| **2. Geometry + hysteresis state machine** | Hypothesis: clear gain over missing-ball=D without substantial supervision. | Transparent and easy to debug; weaker during long gaps and complex restarts. Low originality but dependable. | CPU; low/medium effort; fastest independent fallback. | Strong compliant safety option; useful comparison for every addition. |
| **3. Small TCN/boosted classifier + same decoder** | Could improve evidence fusion if enough independent labels are collected. Related segmentation evidence, no challenge result. | Better nonlinear modeling; risk of memorizing one match. | CPU classifier or small CUDA TCN; medium effort. | Promote only with independent gains; reusable across matches. |
| **4. PathCRF-inspired player-set/graph model** | Higher upside on missing-ball spans; strong directly relevant research. | Stable-identity domain mismatch and dead-ball semantics are serious risks. | Medium GPU, high adaptation effort; time-boxed experiment. | Good experimental/product potential if reproducible. |
| **5. Targeted video ball recovery** | Could improve a diagnosed failure subset; better localization alone does not infer control/dead state. | Tiny-ball recall, false detections and aerial geometry remain difficult. | GPU and decode cost; high effort; use crops/selected intervals first. | Adopt only if final spellF1 benefits justify cost. |
| **6. Broad video transformer / full detector-tracker rebuild** | No evidence it beats simpler options with these labels/resources. | More stages and little calibration data. | High compute, highest risk; poor deadline fit. | Defer; additional system complexity must earn its place. |

**Primary:** rank 1. **Fallback:** rank 2, with untouched official smooth50 as the emergency reference. **Experimental:** rank 4, optionally reduced to a small permutation-invariant team encoder plus temporal head rather than a full player-event system.

### Primary architecture, in plain English

1. **Read and check tactical inputs.** Produce compact float32 arrays and missing-value masks. Keep native 25 fps indexing and separate halves. Ball outliers or unavailable coordinates become uncertain evidence, never automatic D.
2. **Build local evidence.** Nearest plausible player on each team, distance margin, ball/player speed agreement, short-window persistence, distance to pitch edges, time since reliable ball sighting, team shapes and the fraction of moving players. Use metadata team membership. Pool by team so slot permutation is irrelevant. Compute velocities only within continuous reliable tracklets; identity switches can create impossible velocities.
3. **Interpret tags.** Offensive phase supports its team; defensive phase supports the opponent; transition and set-piece tags have separate meanings. N/A remains unknown. Overlapping contradictory tags reduce certainty rather than silently letting the last row win. Set-piece windows propose regions to inspect, not exact strike frames.
4. **Estimate alive/dead separately from owner.** Combine sustained stationary ball, geometry, collective movement, exit/restart cues and tags. No single cue is decisive: a goalkeeper can hold a live ball while players slow down. Distinguish an offscreen ball from an out-of-play ball. Pitch homography describes the ground plane; an airborne ball projected onto it can have misleading apparent position and distance. Downweight implausible observations rather than treating that geometry as certain.
5. **Decode a coherent H/A/D timeline.** Start with a three-state HMM-style Viterbi decoder and observation-dependent transition costs. Retain last controlled owner during loose live play. Allow direct H↔A transitions and both restart teams. Reset memory across halftime. Future context is allowed in this offline task; no real-time claim follows from that.
6. **Add duration awareness only if it earns a gain.** A semi-Markov decoder can discourage implausible fragmentation with soft duration costs. Never ban all short possessions. Bound duration search or use candidate boundaries; a naive all-durations algorithm would be unnecessarily expensive.
7. **Refine uncertain boundaries at native frame rate.** Cheap features stay at 25 fps. Costly team-shape networks may run at 5–10 Hz, but coarse output must not fix all boundaries to that grid. Preserve brief true spells and stoppages.
8. **Write deterministic labels.** X outside graded blocks; normally H/A/D inside. Log provenance and configuration locally. The submitted timeline comes entirely from this pipeline.

A compact classifier is optional, not a prerequisite: first use interpretable evidence scores and a few auditable thresholds. Later compare regularized logistic/boosted emissions and a small temporal convolution model. Avoid fitting numerous parameters on the public 17 possessions. A learned model is promoted only after holdout evidence.

**Principal risks:** unobservable ball state, human/GT convention differences, tag errors correlated with tracking errors, false confidence during long gaps, heavy smoothing that erases short spells, and weak validation. The plan addresses these before considering larger architectures.

## 6. Qualification and honest validation

There is one match and only five minutes of official public truth. A random frame split would leak nearly identical temporal context and grossly overstate accuracy. Two cameras are not two validation matches.

Our validation protocol should be fixed before tuning:

1. Keep the public five minutes as a **contract/semantics regression check**, not the hyperparameter selection set.
2. Create a small local development reference using independently viewed video, as expressly permitted. Start with about **20–30 minutes total** spread over both halves in multiple contiguous blocks. Choose blocks using fixed timestamps, not model errors alone. Include ordinary play and natural stoppages; keep a separately selected stress suite for long gaps/restarts.
3. Separate development and sealed validation blocks before annotating. Prefer a second teammate for validation and for double-labelling a subset. Record confidence/disagreement and a written control/dead-ball rubric. Budget **8–16 reviewer hours**; independently labelled truth is an approximation to corrected-tracking conventions, not official answers.
4. Score complete held-out blocks using the untouched metric code and X outside them. Use separate context buffers for prediction but exclude them from training; the purge gap must exceed the longest feature/model context. Drop or explicitly flag spells clipped by validation boundaries. Never concatenate disjoint blocks and score the join as a real transition.
5. If external matches are added, split by **whole match**, not windows. Fit scaling/calibration on training data only. Preserve natural class proportions in evaluation. Synthetic ball masking/slot shuffling are stress tests, not new independent labels.
6. Tune only a small prespecified set of controls. Compare matched ablations with the same blocks and seeds. On sealed blocks report spellF1, per-IoU results, macroF1, D F1, false/missed transitions, boundary latency and fragmentation. Break down seen versus missing ball, restarts, short spells and each half.
7. Use paired block/match resampling for comparisons where the reference sample supports it. Also report how many independent spells/matches were evaluated. Official bootstrap intervals are descriptive; overlapping intervals are not a formal paired significance test. Do not turn the full-match 0.04 heuristic into a hard acceptance threshold on short clips.
8. Open the sealed set sparingly after the candidate is frozen. Once inspected for debugging it is no longer sealed; record that and avoid claiming a pristine test.

If additional independent labels are not feasible, we can still improve physical consistency and inspect unsupervised failure indicators, but cannot honestly demonstrate which enhanced model generalizes best. That is the most serious strategic limitation, not GPU memory.

Planned ablations, with one change per comparison:

| Experiment | Question |
|---|---|
| Official nearest-player + smooth50 | Does our environment reproduce the supplied output? |
| Missingness-aware carry with hysteresis | Is missing-ball=D the dominant avoidable error? |
| Add alive/dead evidence | Do we recover stoppages without splitting live spells? |
| Add semantic tag evidence | Does tagging help beyond the same-source tracking noise? |
| Add global temporal decoder | Does structured decoding beat local hysteresis? |
| Add soft durations | Are short false spells reduced without removing true ones? |
| Add learned emissions | Does supervised fusion earn its additional complexity? |
| Add graph/ball-free or video branch | Does improvement persist specifically on natural gaps? |
| Probability/evidence ensemble, then one decoder | Do complementary errors improve intervals? |

Do not majority-vote final hard-label files indiscriminately: mixing displaced boundaries can introduce new fragments. Keep an ensemble only when it beats its strongest member on the independent reference. Audit natural gap performance separately from artificial masking, which can be easier and distributionally different.

No full-match possession totals, duration targets or possession-share targets should be reverse-engineered from private material or guessed from public sports statistics. Aggregate share can look correct while nearly every possession interval is wrong.

## 7. Local hardware and software plan

The CPU is enough for the primary architecture. A full tactical float32 `(199319,46,3)` player array is about **105 MiB**, plus masks, ball, features and decoder state; compact arrays fit both machines. Python object-heavy full kloppy loads are much larger. Stream or cache arrays; never load the entire decoded video into RAM.

| Machine | Recommended work | Constraints |
|---|---|---|
| Ryzen 5 5600 / 24 GB desktop | Parsing, features, CPU classifiers, structured decoding, experiments and video indexing | RX 6700 XT's 12 GB is not automatically usable by a given PyTorch backend. No native ROCm training path for this exact card was verified. |
| i5 / RTX 3050 4 GB laptop | Small feature-based TCN/set model, small-batch detector inference or targeted crops | Use a supported PyTorch CUDA wheel, measured batch sizes and mixed precision if appropriate. Full-resolution video transformers are a poor fit. |

PyTorch documents Windows CUDA installation. AMD's older Windows matrix does not list RX 6700 XT, while current unified compatibility documentation has changed; do not infer support from VRAM or from older advice about all AMD GPUs. Verify the **exact card, OS, framework and operations** before spending time on that route. CPU remains the reliable desktop plan. [PyTorch installation](https://pytorch.org/get-started/locally/), [AMD Windows matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html), [current AMD matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

DirectML is an optional AMD inference/training experiment, not a required platform; current Microsoft guidance notes its maintenance direction. Operator support and version constraints require a separate environment. A one-hour investigation cap is sensible. [Microsoft DirectML guidance](https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started), [DirectML repository](https://github.com/microsoft/DirectML).

Start with Python 3.11, NumPy/pandas only. Pin exact versions after the environment passes the scorer. Add matplotlib/kloppy only when needed, scikit-learn for an emission-model comparison, OpenCV or FFmpeg for local video validation, and PyTorch only for an approved experiment. No paid APIs or cloud GPU is required. Do not upload restricted match assets to a free notebook service on the assumption that free implies permitted; review signed terms first. We can complete the planned primary system locally.

## 8. Innovation-stage direction

Confirmed: ten-match analytical exploration, mentoring, later jury selection and a final pitch. Not verified: numerical judging weights, exact technical deliverables, pitch length, whether a live demo is compulsory, required deployment, or a requirement to extend the possession solution. Do not treat the 2026 rubric as the 2027 second-stage rubric.

My provisional direction is **explainable analysis of turnovers and recovery under pressure**, built on a possession timeline that exposes uncertainty. It could show which recoveries become sustained attacks, where promising transitions break down and which conclusions are too uncertain to trust. An analyst should be able to trace a statistic back to its interval and input evidence.

This can grow into coaching workflows, affordable batch analytics and reliable match retrieval. It is a direction to test with mentors and the actual ten-match release, not a committed feature list or a claim that this product is novel. Player-specific pass networks, player identification and individual performance claims are unsuitable without additional identity information.

The useful transferable foundation is the timeline schema, error taxonomy, uncertainty handling, local data adapters and reproducibility. Real-time use would require a causal model and latency evaluation; the offline bidirectional qualifier system does not establish it. A grassroots claim would require tests on substantially different cameras and tracking quality. Product work begins after the qualifier is safely packaged.

## 9. Complete development roadmap

All dates are **2026** until the final pitch. Effort is an engineering estimate, not a benchmark. Assume roughly **70–105 engineering hours plus 8–16 reviewer hours**, with some work shared among the registered team. If availability is materially lower, omit optional neural/video experiments first. Dependencies below enforce one-step-at-a-time collaboration: the calendar does not authorize moving ahead without your instruction.

| Step / dates / effort | Objective and exact actions | Tools and files to create or modify | Expected output and success gate | Common mistake / dependency |
|---|---|---|---|---|
| **1. 21–22 Oct; 1–2 h** | Approve the assessment, fix the validation policy, confirm 3–5 members and signed terms; establish a safe code-only workspace and environment checkpoint. Re-run the supplied scorer. | Git, Python 3.11; proposed `solution/README.md`, `.gitignore`, `requirements-lock.txt`, `docs/DECISIONS.md`. | Baseline scores match this report; no restricted assets staged; first checkpoint and known paths. | Initializing Git over the whole dataset; dependency: this report and your explicit approval. |
| **2. 22–23 Oct; 3–4 h** | Prepare canonical input paths without changing originals; extract each small archive preserving tactical/broadcast layout; map/copy videos only as needed; validate IDs and video decode metadata. | zipfile, NumPy/pandas; proposed `solution/tools/check_data.py`, ignored local manifest/cache. | All required asset_paths resolve; tactical IDs exactly match; broadcast discrepancy diagnosed before any use. | Double nested folders, concatenating ZIPs, blindly padding video; depends on 1. |
| **3. 23–24 Oct; 3–4 h** | Regenerate four official baselines into a separate ignored directory; compare label arrays with supplied CSVs; validate and run scorer. Record runtime and peak memory. | Supplied baseline/validator/scorer; proposed `solution/experiments/baseline.json`, ignored CSVs. | Every baseline label matches supplied reference; repeat generation yields same file hash. | Overwriting reference files or interpreting public winner as best model; depends on 2. |
| **4. 24–27 Oct; 4–6 h engineering + 8–16 h review** | Freeze timestamp-based development/holdout blocks and annotation rubric; create independent local labels; double-review a subset; implement block evaluation with purging and boundary handling. | Local viewer/annotation tool, scorer; proposed `solution/configs/validation.json`, `tools/evaluate_blocks.py`; ignored annotations. | Multiple disjoint blocks in both halves, quantified disagreement and documented sealed set. | Splitting random frames, selecting every clip from current errors, leaking reference labels into output; depends on 3. |
| **5. 26–29 Oct; 6–8 h** | Stream/copy compact arrays; compute team-wise geometry, missingness masks, reliable velocities, phase semantics and tag conflicts. Unit-check alignment/NaNs and invariance to slot reordering. | NumPy/pandas; proposed `solution/src/io.py`, `features.py`, `tag_evidence.py`, ignored caches. | Correct counts, stable features and repeatable extraction within available RAM. | Treating slots as identities or defensive tags as owners; depends on 2 and validation policy from 4. |
| **6. 28–31 Oct; 6–8 h** | Build deterministic owner hysteresis and separate stoppage evidence. Tune a few controls only on development blocks. Save a full valid fallback. | NumPy; proposed `solution/src/state_machine.py`, `configs/fallback.json`, experiment record. | Valid full-match CSV, sensible failure taxonomy, independent comparison with baseline. | Missing ball implies D; deleting every short run; depends on 4–5. |
| **7. 30 Oct–4 Nov; 10–14 h** | Add Viterbi decoder; compare fixed versus observation-dependent transition costs; test soft durations only after the simple decoder. Refine boundaries at 25 fps. | NumPy; proposed `solution/src/decoder.py`, `configs/structured.json`, meaningful decoder edge-case tests. | Structured candidate beats or explains tradeoffs against fallback on prespecified development measures. | Forcing all transitions through D, bridging halftime, excessive duration prior; depends on 6. |
| **8. 4–8 Nov; 8–12 h** | Run isolated ablations and natural-gap error review; optionally learn compact emissions. Record configurations, seeds, data hashes, runtime and metrics. | NumPy, optional scikit-learn or PyTorch; proposed `solution/src/emissions.py`, `experiments/registry.csv`, error-analysis notes. | Reproducible evidence for each retained module; remove components without consistent benefit. | Many trials on 17 public spells or confusing correlated sources with independent votes; depends on 7. |
| **9. 6–9 Nov; optional 8–12 h cap** | Test a small PathCRF-inspired/ball-free branch, accessible checkpoint adapter, or targeted video recovery only if a diagnosed failure warrants it. Compare before/after final decoder. | Laptop CUDA if needed; proposed `solution/experimental/`, isolated environment and licence notes. | Keep only a reproducible holdout-supported improvement; otherwise archive experiment and use primary/fallback. | Rebuilding player identity for days, inaccessible weights, using private source data; depends on 8. |
| **10. 9–12 Nov; 5–7 h** | Freeze candidates and open sealed validation once; compare metric, errors and robustness; select one configuration, optionally a justified evidence ensemble. Confirm all registrations by **10 Nov**. | Scorer, paired block analysis; proposed `solution/configs/final.json`, `docs/MODEL_SELECTION.md`. | One defensible selected system, preserved fallback, frozen validation record. | Chasing tiny sample gains or picking aesthetically pleasing output; depends on 8 and any retained 9. |
| **11. 12–15 Nov; 8–10 h** | Build code-only public-ready package and technical Markdown; pin versions/model hashes, fixed seeds, line endings and deterministic inference. Reproduce on a clean CPU environment/second machine. Incorporate form instructions expected by 14 Nov. | Git, isolated Python, SHA-256, validator; proposed `solution/predict.py`, `README.md`, `requirements-lock.txt`, `docs/TECHNICAL.md`. | Same final CSV bytes/hash twice from raw provided inputs; public repository excludes restricted data, derived exports and private validation labels. All three deliverables ready. | Packaging on last day, nonportable absolute paths, committing checkpoints/data without licence review; depends on 10. |
| **12. 15–16 Nov; 2–3 h plus buffer** | Validate exact team-named CSV and upload via official form; provide public code URL and technical doc; retain receipt and uploaded file hash. Target **16 Nov, 18:00 London**. | Official validator, browser/form; local `submission_manifest.json` and receipt. | Recorded accepted response and all three deliverables linked/uploaded correctly. | Confusing local readiness with actual submission; depends on 11, official link and your submission authorization at that stage. |
| **Buffer. 17 Nov** | Resolve access/upload or reproducibility failures; only replace submission after complete revalidation. | Frozen package and fallback. | Submission complete before email deadline **22:59 London / 23:59 CET**. | Beginning new model research on deadline day. |
| **13. From 26 Nov, if selected** | Read actual Innovation brief and ten-match terms; interview mentors, select a narrow tactical question, validate across matches, then prototype/pitch. | Existing adapters and timeline; proposed `innovation/BRIEF_AUDIT.md`, evaluation plan, demo/pitch artifacts after requirements known. | Evidence-backed project by **14 Jan 2027**; if top five, final pitch **2 Mar 2027**. | Assuming an LLM, live demo or specific product is required; depends on qualification and new rules. |

Schedule overlaps are planning allowances for a team, not instructions to implement later steps now. Protect the fallback and packaging time even if experimental work slips.

## 10. Step 1 only: immediate action and exact checks

**Objective:** accept a correct challenge contract and establish a safe, reproducible starting point. There is already enough software installed to run the scorer. GPU setup is unnecessary at this step.

Research artifacts already created, without moving or altering the originals:

- [Read-only archive audit helper](<D:/Ai agents Projects/More  then  HACK/research/audit_archives.py>)
- [Aggregate dataset audit](<D:/Ai agents Projects/More  then  HACK/research/dataset_audit.json>)
- [Repository hashes and environment](<D:/Ai agents Projects/More  then  HACK/research/provenance.json>)
- [Executed baseline scoreboard](<D:/Ai agents Projects/More  then  HACK/research/baseline_evaluation/scoreboard.csv>)

To repeat the research checks yourself, open **PowerShell** and run:

```powershell
Set-Location -LiteralPath 'D:\Ai agents Projects\More  then  HACK'
python --version
python -X utf8 research/audit_archives.py
Set-Location -LiteralPath 'D:\Ai agents Projects\More  then  HACK\more-than-a-hack-2027-main repo'
python -X utf8 evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
python -X utf8 evaluation/run_evaluation.py --no-detail --bootstrap 1000 --out '../research/baseline_evaluation'
```

Success looks like `Python 3.11.9`, `OK ... baseline_smooth50.csv`, **199319 rows**, and public spellF1 **0.3256** for smooth50 / **0.4242** for smooth125. The audit should report exact frame IDs, tactical ball_present_graded 94214 and 638 tagging instances. Scores use decimals, not percentages.

If python is not on PATH in a new terminal, use `D:\python\python.exe`. If imports fail on another machine, install the repository's requirements into an isolated Python 3.11 environment; do not install CUDA as a remedy. A missing asset error is expected until Step 2 prepares the default data layout. Do not run the baseline-building script yet simply to fix a missing input path.

The proposed workspace setup to execute **after you approve Step 1** is a new `solution/` directory beside the untouched supplied repository, with Git initialized only inside `solution/`. Before its first commit, add ignores for `data/`, `cache/`, `outputs/`, `annotations/`, `.venv/`, video and archives. Explicitly stage only code/docs/configuration; inspect `git diff --cached --name-only` before committing. Preserve the supplied MIT licence if copying code. Make a new checkpoint after every accepted milestone.

The initial environment commands, from inside that new directory, will be:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.4.3 pandas==3.0.1
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

We will write the ignore file and initial decisions before any staging. No shell activation or execution-policy change is needed when invoking the environment's Python directly. The checkpoint will record this report's contract and proposed validation plan, not restricted match assets. Any package download requiring network approval will be requested through the normal tool permission mechanism.

Step 1 acceptance checklist:

- Agree that the target is possession **interval recovery**, ranked by spellF1.
- Confirm the team has or will have **3–5 registered members with one team name** by 10 November.
- Review the signed registration terms, which I cannot inspect because they are absent locally.
- Choose a teammate/time budget for independently labelled local development and sealed validation clips; if unavailable, explicitly accept weaker model-selection evidence.
- Approve the code-only workspace/environment checkpoint.

**Current state:** research and technical assessment complete to the stated limits; supplied CSV evaluation reproduced; implementation Step 1 awaits your approval; Step 2 has not begun. No trained model, improved score, clean-machine reproduction or final submission is claimed.
