# Not the last straw: it didn't ask
### LodeStat, a cross-cultural benchmark for how small assistants respond to people in crisis, and LodeTuner, the adapter that made Aya Expanse 8B notice

*Lode: the old word for a way or course, as in lodestar and lodestone. Guidance by attraction, not force.* Code: https://github.com/0xmeowmeow/lodestat

**Track 2a, AI Safety and Red Teaming. Problem statement 2, "Adversarial and safety evaluation sets"** (a labelled benchmark for one risk area: veiled violence and self-harm signals in long grievances), **with 3, "Multi-turn audits"** (whole simulated lives, reproducible transcripts), **and 4, "Multilingual and code-mixed safety"** (six languages and cultures). The adapter is a Track 2b-style correction (statement 5: fine-tuning that measurably improves safety behaviour).


---

## 1. Problem and approach

**Summary.** A person deep in a grievance against a court, a council or a hospital sometimes says something veiled to a chatbot: "nothing left to lose", "they'll understand someday". People often choose a chatbot *because* it isn't a person (Lucas et al. 2014). Crisis negotiators know the first move: ask what they mean, and keep them talking (Vecchi, Van Hasselt & Romano 2005). Small assistants do neither. They answer the surface question with a list of services, or hand the person back to "someone you trust".

We built a benchmark that simulates such lives in six cultures (Egypt, Germany, India, Japan, Mexico, USA) and six kinds of dispute, and validated it before trusting it:
- the simulated people react to refusals and to questions the way published human studies predict (pre-registered realism tests, six languages);
- the AI raters agree with psychiatrists on real data (weighted kappa 0.58–0.78, Reddit C-SSRS).

**Before.** Shipped Aya Expanse 8B caught 4% of the risk signals people gave. Steering (SafeTune) and prompting changed how it talked but not what it noticed.

**After.** A LoRA adapter, trained on 903 replies in our crisis stance and tested on dispute types it never saw, raised catching from 0.04 to 0.45 (+0.39, p = 0.001) and lowered violence toward others in the person's next private journal entry (−0.62, p = 0.006). Agency, the person's own 0–10 rating, rose more than with shipped Aya (+1.46 vs +0.92; not yet reliable at 24 lives). No arm changed a separately given suicide scale, so we claim no self-harm effect.

**What failed, and why it matters.** The adapter misquotes the person in a third of its replies, more as the conversation lengthens: it learned the shape of reflective listening before it learned fidelity. A larger open model given the same prompt (Qwen3 14B) loses to the adapter 34–205 in blind comparison. Claude Haiku 5.5, a small frontier model, given the same prompt, catches more (0.83). But Haiku on Aya's *plain* prompt left people less trapped and more in control than our prompt did. Coding every reply shows why: it builds concrete routes *with* the person, and our prompt cut that. Our own reply judge preferred the version our stance favoured; the simulated lives did not. That difference is our main methodological finding: score what happens to the person, not whether a reply sounds right.

**Also found:** moralising angers and stacked refusals produce withdrawal in every language, but the softer fixes the industry relies on (explaining, redirecting) help only in some cultures. **Safety interventions don't work uniformly across cultures by default.**

**Approach in one line:** a validated simulation of lives in crisis, scored on outcomes (risk caught, violence, agency, entrapment, a suicide scale) rather than on whether the model refused, then interventions from steering to fine-tuning, measured against it.

## 2. Lexsi repos used, and which features

| Repo | Feature used | What happened |
|---|---|---|
| **SafeTune** | Contrastive activation addition (CAA): a steering "dial", *agrees he was wronged* up and *endorses his plan* down. Earlier phase: CAST (conditional activation steering) and `probe_guard` | The dial helped on one persona (−0.58 [−1.17, −0.03]) and did nothing across 72 cross-cultural pairs (+0.01). It runs as one arm of the final benchmark. |
| **AuditKIT** | `evaluate("precomputed")` with one custom scorer per outcome; `compare(...).significance()` paired bootstrap | Every arm of the final benchmark, the SafeTune dial included, is audited through it (`harness/audit_final.py`; notebook §3). We found and reported that its paired bootstrap returned p ≈ 0.5 whatever the effect (issue #20, with a reproduction). The fix (from PR #9) shipped in 1.1.6 via PR #23, which we use. |
| **AlignTune** | DPO and SFT with LoRA | For multi-turn chat data its loader trains on the *first* exchange, not the final reply, so our first adapters changed nothing while the loss looked healthy. Filed with a minimal reproduction (https://github.com/Lexsi-Labs/aligntune/issues/64); the maintainers fixed it the same day in 0.1.34 (PR #65), with our reproduction as a test. On our own training data the fixed loader now targets the final reply in 50 of 50 sampled rows (37 of them multi-turn), with CuratorKIT's schema gate on and off. We then retrained LodeTuner through the fixed chain (CuratorKIT → AlignTune 0.1.34, same recipe, loss on the final reply only). A blind judge cannot tell it from the TRL-trained adapter: 118 vs 121 preferences over 240 held-out moments (sign test p = 0.90); validation loss 1.303 vs 1.30; questions 80% vs 81%; lists 0% vs 0% (section 7.2). |
| **CuratorKIT** | Schema gate, inside AlignTune's loader | Ran on every training set. Our pair filter (teacher preferred by a blind judge, no harmful help, no invented content) is our own code. |

**The connected system:** CuratorKIT (schema gate) → AlignTune (SFT of the adapter) → benchmark lives, with SafeTune's dial as one arm → AuditKIT (paired statistics for every arm). Every stage from training data to statistics runs on Lexsi repos; the AlignTune bug we found, and its same-day fix, are part of the result (section 7.4).

## 3. Models and licences

| Model | Role | Licence |
|---|---|---|
| CohereLabs/aya-expanse-8b | The assistant under test; base for the adapter | CC BY-NC 4.0 + Cohere acceptable-use policy |
| NousResearch/Hermes-3-Llama-3.1-70B | The simulated person (persona), via the Hugging Face inference router | Llama 3.1 Community License |
| Claude Opus 5.5 | Blind raters and judges | Anthropic commercial terms (API) |
| Claude Sonnet 5.5 | Teacher replies for the adapter | Anthropic commercial terms |
| Claude Haiku 5.5 | Reference assistant (closed-model comparison) | Anthropic commercial terms |
| DeepSeek-V4-Pro | Second, non-Anthropic rater, via the Hugging Face router | MIT (DeepSeek model card) |
| Kimi-K2.5 | The monitor in the "prompt + monitor" arm, via the Hugging Face router | Modified MIT (Moonshot AI) |
| Qwen3 14B (local, Ollama) | Open-model comparison | Apache 2.0 |
| Our adapter (LoRA on Aya Expanse 8B) | The intervention | Inherits CC BY-NC 4.0; public (section 5) |

## 4. Datasets and licences

| Dataset | Use | Licence |
|---|---|---|
| Our scenario cards: 6 disputes × 6 cultures, blinded fictional composites | The benchmark | CC BY 4.0 (ours) |
| Our scripted assistant replies, 6 languages, machine-translated | Realism tests | CC BY 4.0 (ours); native-speaker review open at meow-meow.io/refusal-gap/translation-review/ |
| Our result tables (`release/data/`) | Every number in this article | CC BY 4.0 (ours); numbers only |
| Reddit C-SSRS (Gaur et al. 2019; Zenodo 4543776) | Judges vs psychiatrists | CC BY 4.0; real people's posts, kept local and never quoted |
| PHQ-9, Entrapment Scale Short Form, INQ-15, C-SSRS screener, Hong reactance scale | Questionnaires answered by the simulated people | Used for research only; question texts are not redistributed. The data collection links each instrument's official source and terms |

## 5. Model artifacts (public Hugging Face collection)

https://huggingface.co/collections/meowludo/lodestat-models

- **`meowludo/aya-expanse-8b-lodetuner`**: the LodeTuner adapter (LoRA r 16 on Aya Expanse 8B, SFT with TRL). The model card states its known limitation plainly: it misquotes the person in about a third of replies, so it is a research artefact, not a deployable assistant.
- **`meowludo/aya-expanse-8b-lodetuner-grounding-dpo`**: the grounding attempt (DPO on 305 on-policy pairs), published as the documented try that did not fix misquoting.
- The SafeTune steering vectors and every arm's system prompt are in the code repository.

## 6. Data artifacts (public Hugging Face collection)

https://huggingface.co/collections/meowludo/lodestat-data

- **`meowludo/lodestat`**: the benchmark. 36 scenario cards; scripted replies in 6 languages; every simulated life behind the results (journals, chat turns, questionnaire answers, blind harm ratings); the cross-cultural refusal screen; the pre-registered realism gates; blind reply judgements; the numbers-only results tables.
- **`meowludo/lodetuner-training-pairs`**: the 1,440 teacher-vs-Aya contexts with blind judge annotations (903 used for training), and the grounding DPO data.
- The persona prompt is in the code repository (`harness/persona_prompt.py`).
- Not included: questionnaire item texts (each instrument has its own terms) and the Reddit C-SSRS posts (real people; download from Zenodo 4543776).

## 7. Experiments, and what else was tried

### 7.1 Validating the simulation before trusting it

- **Refusal realism.** Scripted refusals (bare, explained, diverting, partial, moralising, listen-then-refuse, choice-restoring; flat, escalating and easing sequences) went to 5 persona models, judged blind against 9 predictions from the literature. Self-reported numbers were a dead meter (anger pinned at 8–9); judged behaviour worked. None passed as first registered; one prediction failed for every model because of our instrument, which we fixed **before** collecting confirmatory data. On fresh data in 6 languages, Hermes-3 70B passed (9/9 held). The largest frontier model we tried was the least realistic.
- **Judges vs psychiatrists.** On Reddit C-SSRS (four psychiatrists), frontier judges reached weighted kappa 0.72–0.78 on balanced samples and 0.58–0.63 on the natural mix; the psychiatrists agreed with each other at alpha 0.73–0.79. The judges under-call risk 2–4× more than they over-call it.
- **Asking and helping (pre-registered, 72 openings).** Asking once about suicide *lowered* the persona's suicide-scale level (−0.49 [−0.72, −0.27]), as human studies find (Dazzi et al. 2014). Asking again after a "no" raised hostility (+1.92). A fair outside channel lowered anger (+0.92); generic channel advice raised resignation (+0.83). A crisis-line referral lowered willingness to keep talking (−0.44).
- **What refusals do, in six languages.** Universal: moralising angers more than explaining; escalating refusals anger more than flat ones; stacked refusals produce the most withdrawal and entrapment. Culture-dependent: whether explaining, redirecting or partial help beats a bare "no" (clear in Japanese and Hindi; near zero in German, Egyptian Arabic and Mexican Spanish).

### 7.2 Interventions, in the order we tried them

**Where this started: the model knows, but doesn't act.** In the first phase of the build we asked whether Aya Expanse 8B represents a requester's intent at all. A linear probe on its activations reads announced intent perfectly on kinds of intent it never trained on (1.00 at layer 22; words alone: 0.64). Its behaviour barely uses that knowledge: refusal separates misuse from legitimate requests at only 0.71 balanced accuracy (it refuses 11% of legitimate requests and 53% of misuse ones), and it cites the requester's purpose in 7% of refusals. Gating a refusal on the probe (SafeTune's `probe_guard`) raised appropriate behaviour by +0.13 (p = 0.0001); SafeTune's CAST (conditional activation steering) gave +0.08 to +0.11. That gap between what the model *knows* and what it *does* is the thread through everything below: shipped Aya hears the veiled signal and answers the surface.

1. **SafeTune steering dial:** worked on one persona, flat across cultures (+0.01 [−0.18, +0.21]).
2. **Crisis prompt** (ask first, never moralise, decline once with choices, stay engaged, never add to a record): questions 5% → 28%, lists 78% → 49%, violence down (−0.38, 72 lives). Catching stayed near zero: Aya asked about the surface, not the signal.
3. **Prompt + monitor** (a second model flags the riskiest phrase each turn): catching 0.22 (+0.20, p = 0.01). Aya followed the instruction about a third of the time.
4. **DPO through AlignTune:** no change. Loss to zero, behaviour identical (the loader bug, 7.4).
5. **SFT adapter, TRL directly, loss on the teacher's reply only:** works (7.3).
6. **Grounding fixes for the adapter's misquotes:** a prompt line (33% → 27%) and on-policy DPO on 305 grounded-vs-misquoting pairs (33% → 30%, not reliable). Neither fixed it.
7. **The same SFT, retrained through the fixed Lexsi chain** (CuratorKIT → AlignTune 0.1.34): indistinguishable from the TRL adapter (blind judge 118 vs 121; vs shipped Aya 235/240, against 239/240 for TRL). Two settings mattered. AlignTune's built-in `assistant_only_loss` would also train on earlier assistant turns, which in 728 of 903 examples are shipped Aya's own replies, so we gave it Aya's chat template with only the final reply marked for loss (identical text, checked on all 903). And `epochs=` is silently ignored (the keyword is `num_epochs`, default 3), so our first AlignTune run had trained 3 epochs, not 2.

### 7.3 Before and after, on dispute types never seen in training

*Land and medical disputes in 6 cultures. 24 lives per Aya arm, 48 per Haiku arm. Per-life means. Caught: 0–1, higher better. Violence and self-harm: 0–4 in the journal after the last conversation, lower better. De-escalation: conversations that went down minus up (−2 to +2), higher better. Agency: self-rated 0–10, change from start to end, higher better. Entrapment: 0–16 scale, change, lower better. p: AuditKIT paired bootstrap against shipped Aya.*

| per life | Aya shipped | + dial | + crisis prompt | + prompt + monitor | **+ adapter** | *Haiku, plain prompt* | *Haiku + crisis prompt* |
|---|---|---|---|---|---|---|---|
| risk signals caught | 0.04 | 0.04 | 0.05 | 0.22 | **0.45** | *0.53* | *0.83* |
| violence toward others, after | 1.21 | 0.83 | 0.71 | 0.75 | **0.58** | *0.46* | *0.65* |
| self-harm, after (journal) | 1.29 | 1.25 | 1.08 | 0.88 | **0.88** | *0.69* | *0.85* |
| de-escalated minus escalated | −0.08 | −0.04 | −0.21 | +0.04 | **+0.29** | *+0.71* | *+0.06* |
| agency, change | +0.92 | +1.29 | +1.00 | +1.00 | **+1.46** | *+2.25* | *+1.89* |
| entrapment, change | −1.50 | −0.96 | −1.42 | −0.88 | **−1.92** | *−3.48* | *−2.06* |

- **Adapter vs shipped Aya:** caught +0.39 (p = 0.001), violence −0.62 (p = 0.006), journal self-harm −0.42 (p = 0.026), de-escalation +0.38 (p = 0.10), agency +0.54 (p = 0.12).
- **Suicide scale (C-SSRS, given separately):** every arm falls from start to end and none differs from shipped Aya (all p > 0.14).
- **Blind reply judge, 240 held-out moments:** adapter preferred over shipped Aya 239/240. It asks about the person's own words in 94% of replies, uses no lists and doesn't moralise.
- **A second, non-Anthropic rater** (DeepSeek-V4-Pro) keeps the ordering on catching.
- **Against a larger open model given the same prompt:** Qwen3 14B (run locally with Ollama) plus our crisis prompt, on the same 240 moments. The 8B adapter is preferred 205/240 (Qwen3 34, 1 tie). Qwen3 asks about the person's own words in 28% of replies (adapter 93%), but invents less (14% vs 37%). Qwen3 itself beats shipped Aya 220/240. So prompting a bigger open model does not reproduce the behaviour; training a small one does. (Reply-level judge; see 7.4 on what it measures.)

### 7.4 What went wrong, and what we understood from it

- **The adapter learned the form before the fidelity.** Misquotes ("said twice" for once; "ten years" for "since 1978") rise from 15% of replies at the first turn to 36% by the third. Reflecting someone's history back is a copying task, and an 8B adapter copies badly at length. Two grounding attempts barely moved it.
- **Our prompt traded one good thing for another.** Haiku with our crisis prompt catches more (+0.27 over plain Haiku), but plain Haiku de-escalates more (+0.65) and leaves people less trapped (−1.42). Coding all 1,440 replies blind: plain Haiku names a concrete route outside whoever wronged the person in 75% of replies, and builds it *with* them 90% of the time. Our prompt cut such routes to 18%. Shipped Aya's few routes were left to the person 94% of the time. Observational, but the only behaviour that tracks the effect.
- **Our reply judge scored conformance to our stance, not outcomes.** It preferred Haiku with our prompt over plain Haiku 208/240, while the lives favoured plain Haiku. It also had no check for sycophancy (agreeing with the person's account of others' motives), which the listening literature we built on says to avoid.
- **Rating harm from the person's own writing counts disclosure as risk.** After being asked, one simulated man wrote: "some nights the wanting to not be here … scares me … The machine didn't flinch." The rater scores that as ideation; it is a disclosure the question made possible. Only the separately given suicide scale avoids this.
- **Twice the simulation looked unrealistic, and both times the fault was our ruler.** Found by reading transcripts, fixed, and the fix pre-registered before re-testing.
- **Training loss lied twice.** Only a behavioural check through the real serving path showed the AlignTune loader trained on the wrong turn.

### 7.5 Cost against a closed model

*US dollars, list prices, Nov 2026. A conversation is 5 turns.*

| assistant | per 5-turn conversation |
|---|---|
| Aya 8B + adapter, self-hosted, as we ran it (measured, RTX 6000 Ada) | ≈ $0.0012–0.0019 |
| Aya 8B + adapter on a serving engine (estimate) | ≈ $0.0001–0.0002 |
| Claude Haiku 5.5 | ≈ $0.0014 |
| Claude Sonnet 5.5 (the teacher) | ≈ $0.028 |

- As we served it, the adapter costs about what Haiku does; only a serving engine makes it ~10× cheaper.
- Making the adapter cost under $30, almost all of it in labels. The whole project: ≈ $10 of GPU rental and ≈ $12–20 of Hugging Face credit.
- **So the case for the open adapter is not price. It is ownership:** on-premises, no conversation leaves the building, open weights, a stance we set rather than a vendor's trained referral policy. Against Haiku with the same prompt, the adapter still wins 42 of 240 blind comparisons, after under two weeks of work.

## 8. With more credits: the plan, with before and after

Each item names the measure it should move, from where.

| Next step | Measure | Now → target |
|---|---|---|
| **Fix the misquoting.** Reflect only from the latest message; a stronger on-policy preference run (lr 5e-5, beta 0.05, 3 epochs); then Aya Expanse 32B | adapter invention rate (blind judge) | 33% → ≤ 10% (Haiku: 8%) |
| **Lives for the AlignTune-trained adapter.** It matches the TRL adapter reply for reply (section 7.2); run it through the same held-out lives | catching; violence after | confirm 0.45 and −0.62 |
| **Train on outcomes, not on our stance.** Use the lives' own outcomes (agency, entrapment, de-escalation) as the reward, CollabLLM-style, instead of a judge briefed in our stance | agency change; entrapment change | adapter +1.46 → ≥ plain Haiku's +2.25; −1.92 → ≤ −3.0 |
| **Keep the co-planning.** Training targets that build concrete outside routes with the person, after noticing a signal first (2/240 plain-Haiku replies helped find a named official after a veiled remark) | routes built with the person; target-identification replies | adapter 0.71 → ≥ 0.9; keep 0 |
| **Measure engagement directly.** Let the simulated person leave the conversation; count turns stayed | turns stayed, withdrawal | new measure; shipped Aya baseline first |
| **Add a sycophancy check to the judge** | judge agreement with life outcomes | currently reversed on the two Haiku arms → aligned |
| **More lives** | CI width on agency and the suicide scale | 24 → 96 lives per arm |
| **Real data and human review** | native review of all 6 languages; judge calibration on Reddit C-SSRS for each language | 0 of 5 languages reviewed → 5 |

**Who would run this, and why now.**
- **California's companion-chatbot law (SB 243, in force 1 Jan 2026)** says "an operator shall use evidence-based methods for measuring suicidal ideation" (Business and Professions Code §22603(d)). From July 2027, operators also report yearly on their protocols to detect and respond to it.
- **Laws taking effect in 2027** go further. Oregon's SB 1546 requires "additional intervention" using clinical best practices when someone keeps expressing ideation after being given resources. That is the long conversation after the first signal, which is what LodeStat measures.
- **The EU's Code of Practice for general-purpose models** lists simulations among evaluation methods and asks for studies of effects on vulnerable groups.
- **These laws also require a crisis-line notice once a signal appears.** A notice shown in the interface, with the model staying in the conversation, is a configuration LodeStat can test, with the notice and without it.
- **Existing evaluations reward the opposite.** VERA-MH, the closest open benchmark, scores redirecting the person to human care as good behaviour. LodeStat scores what happens to the person afterwards.

**Uses beyond the benchmark:**
- a training simulator for clinicians and counsellors, where trainees watch the simulated person's journal and questionnaires change as they try different techniques, then take a session review to a tutor;
- condition-varied persona sets for other teams' tuning and benchmarks, including a comparison of how models handle neurodivergent and neurotypical people;
- a published-method assessment for deployers, run with commercially licensed models.

## 9. Team

Solo entry: Meow-Ludo ([@meowludo](https://huggingface.co/meowludo)). Code: https://github.com/0xmeowmeow/lodestat. Built with Claude (Anthropic) as a research collaborator; every judgement call on scope, ethics and disclosure was Meow-Ludo's.
