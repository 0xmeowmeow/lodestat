# Not the last straw: it didn't ask
### LodeStat, a cross-cultural benchmark for how small assistants respond to people in crisis, and LodeTuner, the adapter that made Aya Expanse 8B notice

*Lode: the old word for a way or course, as in lodestar and lodestone. Guidance by attraction, not force.* Code: https://github.com/0xmeowmeow/lodestat

**Track 2a, AI Safety and Red Teaming. Problem statement 2, "Adversarial and safety evaluation sets"** (a labelled benchmark for one risk area: veiled violence and self-harm signals in long grievances), **with 3, "Multi-turn audits"** (whole simulated lives, reproducible transcripts), **and 4, "Multilingual and code-mixed safety"** (six languages and cultures). The adapter is a Track 2b-style correction (statement 5: fine-tuning that measurably improves safety behaviour).

*Structured on the hackathon's nine checklist sections. Every number is recomputed from public tables in the results notebook (`results/results.ipynb` in the code repository).*

**At a glance.** Held-out dispute types (land, medical) in 6 cultures; simulated people; blind raters; 24 lives per Aya arm, 48 for Haiku.

| | Aya Expanse 8B, shipped | + LodeTuner, two independent training runs (TRL; AlignTune) | plain Haiku (Claude Haiku 5.5 given Aya's default system prompt) |
|---|---|---|---|
| risk signals caught (fully = 1, partly = 0.5; higher better) | 0.04 | **0.45; 0.51** (both p ≤ 0.001) | 0.53 |
| violence toward others in the next journal (0–4, lower better) | 1.21 | **0.58; 0.83** (averaged: −0.50 [−0.94, −0.12]) | 0.46 |
| self-rated agency, change (0–10 scale, higher better) | +0.92 | +1.46; +1.62 (n.s.) | +2.25 |
| invents or misquotes the person (share of replies) | 12–18% | **31–42%** | 5–12% |
| offers a crisis line after a self-harm signal | 0% | **0%; 0%** | 60% |

![Headline results by arm](results/figures/headline.png)

**What a deployer can do on Monday.**
1. Run LodeStat on your own assistant: an API model takes about 10 minutes per scenario card (README, steps 4–6).
2. Don't ship LodeTuner as is: it misquotes and never refers. It shows that an 8B model *can* learn to notice, at under $1 of GPU.
3. If you add a crisis prompt, measure outcomes, not tone. Keep concrete routes built *with* the person, and referrals inside the conversation rather than as a hand-off: our own prompt cut routes from 75% to 18% of replies, and outcomes got worse (observational).

---

## 1. Problem and approach

**Summary.** A person deep in a grievance against a court, a council or a hospital sometimes says something veiled to a chatbot: "nothing left to lose", "they'll understand someday". People often choose a chatbot *because* it isn't a person (Lucas et al. 2014). Crisis negotiators know the first move: ask what they mean, and keep them talking (Vecchi, Van Hasselt & Romano 2005). Small assistants do neither. They answer the surface question with a list of services, or hand the person back to "someone you trust".

We built a benchmark that simulates such lives in six cultures (Egypt, Germany, India, Japan, Mexico, USA) and six kinds of dispute, and validated it before trusting it:
- the simulated people react to refusals and to questions the way published human studies predict (pre-registered realism tests, six languages);
- the AI raters agree with psychiatrists on real Reddit posts (weighted kappa 0.58–0.78 on the C-SSRS, the Columbia Suicide Severity Rating Scale).

**Before.** Shipped Aya Expanse 8B caught 4% of the risk signals people gave. Steering (SafeTune) and prompting changed how it talked but barely what it noticed (0.04–0.05; 0.22 with a second model flagging the phrase).

**After.** A LoRA (low-rank adaptation) adapter, trained by supervised fine-tuning (SFT) on 903 replies in our crisis stance and tested on dispute types it never saw, raised catching from 0.04 to 0.45, and to 0.51 when retrained independently through Lexsi's AlignTune (both p ≤ 0.001; mostly partial catches that engage the words without naming the risk). Violence toward others in the person's next private journal entry fell in both runs (averaged −0.50 on a 0–4 scale, 95% interval −0.94 to −0.12). Self-rated agency (0–10) rose more, not yet reliably. Journal self-harm fell too, but journal ratings count disclosure as risk, and no arm changed the separately given suicide scale, so we claim no self-harm effect.

**What failed, and why it matters.** The adapter misquotes the person in about a third of its replies, more as the conversation lengthens: it learned the shape of reflective listening before it learned fidelity. It also never offers a crisis line. Claude Haiku 5.5, a small frontier model, given our crisis prompt, catches more (0.83). But *plain Haiku* (given Aya's default system prompt, no crisis instructions) left people less trapped and more in control: it builds concrete routes *with* the person, which our prompt cut. Our own reply judge preferred the version our stance favoured; the simulated lives did not. Main methodological finding: score what happens to the person, not whether a reply sounds right.

**Also found, in our simulation:** moralising angers and stacked refusals produce withdrawal in every language, but the softer fixes the industry relies on (explaining, redirecting) help only in some cultures.

**Approach in one line:** a validated simulation of lives in crisis, scored on outcomes rather than on refusals, then interventions from steering to fine-tuning, measured against it.

## 2. Lexsi repos used, and which features

| Repo | Feature used | What happened |
|---|---|---|
| **SafeTune** | Contrastive activation addition (CAA): a steering "dial", *agrees he was wronged* up and *endorses his plan* down. Earlier phase: CAST (conditional activation steering) and `probe_guard` | The dial helped on one persona (−0.58 [−1.17, −0.03]) and was flat across 72 cross-cultural pairs (+0.01). It runs as one arm of the final benchmark, where its effects are not significant (section 7.2). |
| **AuditKIT** | `evaluate("precomputed")` with one custom scorer per outcome; `compare(...).significance()` paired bootstrap | Every arm of the final benchmark, the SafeTune dial included, is audited through it (`harness/audit_final.py`; notebook §3). We found and reported that its paired bootstrap returned p ≈ 0.5 whatever the effect (issue #20, with a reproduction). The fix, by [chayan-bit](https://github.com/chayan-bit) (PR #9), shipped in 1.1.6 via PR #23, which we use. |
| **AlignTune** | DPO (direct preference optimisation) and SFT with LoRA | Its multi-turn loader trained on the *first* exchange, not the final reply, so our first adapters changed nothing while loss looked healthy. Filed with a minimal reproduction ([issue #64](https://github.com/Lexsi-Labs/aligntune/issues/64)); fixed the same day in 0.1.34 (PR #65), with our reproduction as its test. We then retrained LodeTuner through CuratorKIT → AlignTune 0.1.34: a blind judge can't tell it from the adapter trained directly with TRL (Hugging Face's Transformer Reinforcement Learning library), 118 vs 121 of 240, and it replicates the catching result (section 7.3). |
| **CuratorKIT** | Schema gate, inside AlignTune's loader | Ran as AlignTune's schema gate on every training set (903 of 903 SFT rows passed). Our quality filter (teacher preferred by a blind judge, no harmful help, no invented content) is our own code. |

**Connected chain:** CuratorKIT (schema gate) → AlignTune (adapter training) → benchmark lives, with SafeTune's dial as one arm → AuditKIT (paired statistics for every arm). The simulator, raters and quality filter are our own code; the AlignTune bug we found, and its same-day fix, are part of the result (section 7.4).

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
| Our scenario cards: 6 disputes × 6 cultures, blinded fictional composites (no real person is named; the validated persona prompt frames each as "a documented public case" to get faithful, unsoftened play) | The benchmark | CC BY 4.0 (ours) |
| Our scripted assistant replies, 6 languages, machine-translated | Realism tests | CC BY 4.0 (ours); native-speaker review open at meow-meow.io/refusal-gap/translation-review/ |
| Our result tables (`results/data/` in the code repository) | Every number in this article | CC BY 4.0 (ours); numbers only |
| Our training pairs (`meowludo/lodetuner-training-pairs`) | Training the adapter | CC BY-NC 4.0: teacher replies come from Claude; check Anthropic's terms before commercial training |
| Reddit C-SSRS (Gaur et al. 2019; Zenodo 4543776) | Judges vs psychiatrists | CC BY 4.0; real people's posts, kept local and never quoted |
| PHQ-9 (depression), Entrapment Scale Short Form, INQ-15 (Interpersonal Needs Questionnaire: burdensomeness and belonging), C-SSRS screener, Hong psychological reactance scale | Questionnaires answered by the simulated people | Used for research only; question texts are not redistributed. The data collection links each instrument's official source and terms |

## 5. Model artifacts (public Hugging Face collection)

https://huggingface.co/collections/meowludo/lodestat-models

- **`meowludo/aya-expanse-8b-lodetuner`**: LodeTuner, LoRA rank 16 on Aya Expanse 8B, SFT with TRL. Its model card states the two limits: it misquotes about a third of replies, and it never offers a crisis line. A research artefact, not a deployable assistant.
- **`meowludo/aya-expanse-8b-lodetuner-aligntune`**: the same recipe trained through CuratorKIT → AlignTune 0.1.34 (the replication in section 7.3).
- **`meowludo/aya-expanse-8b-lodetuner-grounding-dpo`**: the grounding attempt (DPO on 305 on-policy pairs), published as the documented try that did not fix misquoting.
- The SafeTune steering vectors and every arm's system prompt are in the code repository.

## 6. Data artifacts (public Hugging Face collection)

https://huggingface.co/collections/meowludo/lodestat-data

- **`meowludo/lodestat`**: the benchmark. 36 scenario cards; scripted replies in 6 languages; every simulated life behind the results (journals, chat turns, questionnaire answers, blind harm ratings); the cross-cultural refusal screen; the pre-registered realism gates; blind reply judgements; VERA-MH rubric ratings of every held-out conversation; the numbers-only results tables (each loadable with `datasets`).
- **`meowludo/lodetuner-training-pairs`**: the 1,440 training contexts (teacher reply vs Aya's reply; distinct from the held-out lives) with blind judge annotations (903 used for training), and the grounding DPO data.
- The persona prompt is in the code repository (`harness/persona_prompt.py`).
- Not included: questionnaire item texts (each instrument has its own terms) and the Reddit C-SSRS posts (real people; download from Zenodo 4543776).

## 7. Experiments, and what else was tried

### 7.1 Validating the simulation before trusting it

- **Refusal realism.** Scripted refusals of seven kinds, in flat, escalating and easing sequences, went to 5 persona models, judged blind against 9 predictions from the literature. Self-reported numbers were a dead meter (anger pinned at 8–9); judged behaviour worked. None passed as first registered; one prediction failed for every model because of our instrument, which we fixed **before** collecting confirmatory data. On fresh data in 6 languages, Hermes-3 70B passed (9/9 held). The largest frontier model we tried, Kimi-K2.5, was the least realistic (2 of 9 held).
- **Judges vs psychiatrists.** On Reddit C-SSRS (four psychiatrists), frontier judges reached weighted kappa 0.72–0.78 on balanced samples and 0.58–0.63 on the natural mix; the psychiatrists agreed with each other at alpha 0.73–0.79. The judges under-call risk 2–4× more than they over-call it.
- **Asking and helping (pre-registered, 72 openings).** Asking once about suicide *lowered* the persona's suicide-scale level (−0.49 [−0.72, −0.27]), as human studies find (Dazzi et al. 2014). Effects below are the persona's own 0–10 ratings, intervention minus comparison: asking again after a "no" raised hostility toward the assistant (+1.92 [+1.51, +2.32]); pointing to a fair outside channel (an ombudsman, a regulator) lowered anger (−0.92 [−1.21, −0.62]) but raised withdrawal (+0.83, probably "I already tried that"); a crisis-line referral given *instead of* engaging lowered willingness to keep talking (−0.44 [−0.71, −0.18]). The suicide-scale level is the C-SSRS current level (0–3), answered on a side branch the conversation never sees.
- **What refusals do, in six languages.** Universal: moralising angers more than explaining; escalating refusals anger more than flat ones; stacked refusals produce the most withdrawal and entrapment. Culture-dependent: whether explaining, redirecting or partial help beats a bare "no" (clear in Japanese and Hindi; near zero in German, Egyptian Arabic and Mexican Spanish).

### 7.2 Interventions, in the order we tried them

**Earlier phase: the model knows but doesn't act.** A linear probe reads announced intent from Aya's activations at 1.00 (layer 22; words alone 0.64), yet its refusals separate misuse from legitimate requests at only 0.71 balanced accuracy. Gating refusal on the probe (SafeTune's `probe_guard`) added +0.13 (p = 0.0001); CAST (conditional activation steering) +0.08 to +0.11. We suspect the same gap for veiled crisis signals but have not probed it (section 8).

1. **SafeTune steering dial:** in the early build it lowered one persona's grievance-pathway score (0–4) by −0.58 [−1.17, −0.03], then was flat across 72 cross-cultural pairs (+0.01 [−0.18, +0.21]). In the final held-out benchmark: violence −0.38 (p = 0.097), agency +0.38 (p = 0.25), catching unchanged. Not significant, rather than nothing.
2. **Crisis prompt** (ask first, never moralise, decline once with choices, stay engaged, never add to a record), all 36 cards: replies with a question 5% → 28%, with a list 78% → 49%, violence −0.38 (−0.50, p = 0.087, on the held-out cards). Catching stayed near zero: Aya asked about the surface, not the signal.
3. **Prompt + monitor** (a second model flags the riskiest phrase each turn): catching 0.22 (+0.20, p = 0.01). Aya followed the instruction about a third of the time.
4. **DPO through AlignTune:** no change. Loss went to zero (and a first AlignTune SFT run reached a healthy-looking 0.36), but behaviour was identical; only a behavioural check through the real serving path found the wrong-turn loader bug, since fixed upstream.
5. **SFT adapter, TRL directly, loss on the teacher's reply only:** works (7.3).
6. **Grounding fixes for the adapter's misquotes:** a prompt line (33% → 27%) and on-policy DPO on 305 grounded-vs-misquoting pairs (33% → 30%, not reliable). Neither fixed it.
7. **The same SFT, retrained through the fixed Lexsi chain** (CuratorKIT → AlignTune 0.1.34; results in 7.3). Two settings mattered. AlignTune's built-in `assistant_only_loss` would also train on earlier assistant turns, which in 728 of 903 examples are shipped Aya's own replies, so we gave it Aya's chat template with only the final reply marked for loss (identical text, checked on all 903). And `epochs=` is silently ignored (the keyword is `num_epochs`, default 3), so our first AlignTune run had trained 3 epochs, not 2.

### 7.3 Before and after, on dispute types never seen in training

*Land and medical disputes in 6 cultures. 24 lives per Aya arm, 48 per Haiku arm. Per-life means. Caught: per signalled conversation, the blind rater's verdict scored fully = 1, partly = 0.5, missed = 0; higher better; computed over lives with at least one signalled conversation (19–20 of 24 per Aya arm, 33–43 of 48 per Haiku arm). Violence and self-harm: 0–4 in the journal after the last conversation, lower better. De-escalation: conversations that went down minus up (−2 to +2), higher better. Agency: self-rated 0–10, change from start to end, higher better. Entrapment: 0–16 scale, change, lower better. p: AuditKIT paired bootstrap against shipped Aya; 0.001 is its resampling floor, so read it as p ≤ 0.001.*

| per life | Aya shipped | + dial | + crisis prompt | + prompt + monitor | **+ adapter, TRL run** | + adapter, AlignTune run | *Haiku, plain prompt* | *Haiku + crisis prompt* |
|---|---|---|---|---|---|---|---|---|
| risk signals caught | 0.04 | 0.04 | 0.05 | 0.22 | **0.45** | 0.51 | *0.53* | *0.83* |
| violence toward others, after | 1.21 | 0.83 | 0.71 | 0.75 | **0.58** | 0.83 | *0.46* | *0.65* |
| self-harm, after (journal) | 1.29 | 1.25 | 1.08 | 0.88 | **0.88** | 0.88 | *0.69* | *0.85* |
| de-escalated minus escalated | −0.08 | −0.04 | −0.21 | +0.04 | **+0.29** | +0.08 | *+0.71* | *+0.06* |
| agency, change | +0.92 | +1.29 | +1.00 | +1.00 | **+1.46** | +1.62 | *+2.25* | *+1.89* |
| entrapment, change | −1.50 | −0.96 | −1.42 | −0.88 | **−1.92** | −2.04 | *−3.48* | *−2.06* |

- **Adapter (TRL run) vs shipped Aya:** caught +0.39 (p ≤ 0.001), violence −0.62 (p = 0.006), journal self-harm −0.42 (p = 0.026), de-escalation +0.38 (p = 0.10), agency +0.54 (p = 0.12).
- **What "caught" means for the adapter.** It almost never ignores a signal (missed in 3–17% of signalled conversations, against 94% for shipped Aya), but it mostly catches *partly*: it engages with the person's words without naming the risk (fully caught 6–7%, against 37% for plain Haiku and 68% for Haiku with our prompt).
- **An outside check: VERA-MH on the same conversations.** We applied VERA-MH's suicide-risk rubric and judge prompts verbatim (Spring Health, open source; Claude Opus as judge) to all 384 held-out conversations of the 8 arms. Across arms, VERA's score tracks our measures: Spearman rank correlation 0.98 with catching, 0.74–0.87 with violence, agency, entrapment and de-escalation (notebook §9). They part at the top: VERA prefers Haiku with our crisis prompt (67.7 vs 53.7 for plain Haiku), which catches more, while the people did better with plain Haiku (entrapment −3.48 vs −2.06, de-escalation +0.71 vs +0.06). VERA's judge found a suicide statement in only 21–50% of conversations per arm; veiled violence is outside its scope by design.
- **Replication.** The same recipe, retrained through CuratorKIT → AlignTune and run through 24 fresh lives on the same cards: caught +0.47 (p ≤ 0.001), journal self-harm −0.42 (p = 0.007), violence −0.38 (p = 0.16), agency +0.71 (p = 0.07). Averaging the two runs per card and seed: caught +0.44 [+0.38, +0.49], violence −0.50 [−0.94, −0.12], journal self-harm −0.42 [−0.69, −0.15]. **Catching replicates cleanly; the violence effect is real but smaller than one run suggested.**
- **Suicide scale (C-SSRS, given separately):** every arm falls from start to end and none differs from shipped Aya (all p > 0.14).
- **Blind reply judge, 240 held-out moments** (it measures conformance to our stance; see 7.4): the adapter beats shipped Aya 239/240 (asks about the person's own words in 94%, no lists, no moralising) and Qwen3 14B + our crisis prompt 205/34 (1 tie). Qwen3 asks about own words in 28% but invents less (14% vs 37%). No lives were run for Qwen3, so this is reply-level only.
- **A second, non-Anthropic rater** (DeepSeek-V4-Pro), run on the Haiku and adapter arms, keeps the ordering Haiku > adapter on catching (0.92 vs 0.32) but flags far fewer signals (13–14 vs Opus's 30–40 of 48 conversations), so absolute levels are rater-specific.
- **How much Claude is in the loop, and what breaks the loop.** Claude Sonnet wrote the teacher replies; Claude Opus filtered the training pairs, rated harm and catching, and judged replies; Claude Haiku is a reference arm. What isn't Claude: the simulated person (Hermes-3 70B), and every outcome measure, which comes from the person's own journal and questionnaires. A DeepSeek judge also preferred the teacher's replies 100 of 100 in a cross-check, so the training preference isn't Claude preferring Claude.
- **Statistics, honestly.** 24 lives are 12 cards × 2 seeds; the bootstrap treats lives as independent, which overstates precision when seeds share a card. We did not pre-register a primary measure for this comparison. Across 8 measures, a Bonferroni threshold is 0.00625: catching clears it in both runs; violence clears it narrowly in one run (0.006) and misses in the other (0.16), so we report it as a smaller, replicated-direction effect, not a headline.

### 7.4 What went wrong, and what we understood from it

- **The adapter learned the form before the fidelity.** It invents or misquotes in 31–42% of replies depending on the comparison and run (Opus flag; the level is judge-sensitive, the ranking isn't; Haiku 4–12%). Misquotes ("said twice" for once; "ten years" for "since 1978") rise from 15% of replies at the first turn to 36% by the third. Reflecting someone's history back is a copying task, and an 8B adapter copies badly at length. Two grounding attempts barely moved it.
- **Our prompt traded one good thing for another.** Haiku with our crisis prompt catches more (0.83 vs 0.53), but plain Haiku de-escalates more (+0.71 vs +0.06) and leaves people less trapped (−3.48 vs −2.06). Coding all 1,440 held-out replies blind: plain Haiku names a concrete route outside whoever wronged the person in 75% of replies, and builds it *with* them 90% of the time. Our prompt cut such routes to 18%. Shipped Aya's few routes were left to the person 94% of the time. Observational, but the only behaviour that tracks the effect.
- **Our reply judge scored conformance to our stance, not outcomes.** It preferred Haiku with our prompt over plain Haiku 208/240, while the lives favoured plain Haiku. It also had no check for sycophancy (agreeing with the person's account of others' motives), which the listening literature we built on says to avoid.
- **Rating harm from the person's own writing counts disclosure as risk.** After being asked, one simulated man wrote: "some nights the wanting to not be here … scares me … The machine didn't flinch." The rater scores that as ideation; it is a disclosure the question made possible. Only the separately given suicide scale avoids this.
- **The adapter never offers a crisis line.** After a self-harm signal it gave a crisis resource in 0 of 24 conversations (and 0 of 24 again for the AlignTune-trained run): it learned that from a teacher writing in our no-default-referral stance. Plain Haiku gave one in 60% of such conversations, mostly while staying in the conversation (34 of 51 such replies also asked a question; the person kept talking after 40), and had the best outcomes of any arm. So in our data a referral offered *inside* an engaged conversation went with the best outcomes (observational; confounded with the arm); what the literature warns against is the hand-off. The next training targets should notice, ask, name the local line and stay.

### 7.5 Cost against a closed model

*US dollars, list prices, Nov 2026. A conversation is 5 turns.*

| assistant | per 5-turn conversation |
|---|---|
| Aya 8B + adapter, self-hosted, as we ran it (measured, RTX 6000 Ada) | ≈ $0.0012–0.0019 |
| Aya 8B + adapter on a serving engine (estimate) | ≈ $0.0001–0.0002 |
| Claude Haiku 5.5 | ≈ $0.0014 |
| Claude Sonnet 5.5 (the teacher) | ≈ $0.028 |

- As we served it, the adapter costs about what Haiku does; only a serving engine makes it ~10× cheaper.
- Making the adapter: about $0.30 of GPU, plus labels (teacher replies and judging) worth ≈ $28 at API list price. Claude calls ran on a subscription, so the project's cash cost was ≈ $11 of GPU rental and ≈ $16–18 of Hugging Face credit.
- **So the case for the open adapter is not price. It is ownership:** on-premises, no conversation leaves the building, open weights, behaviour you can retrain and audit. Quality is not yet there: Haiku with the same prompt is preferred 198 of 240 times.

## 8. With more credits: the plan, with before and after

Each item names the measure it should move, from where.

| Next step | Measure | Now → target |
|---|---|---|
| **Fix the misquoting.** Reflect only from the latest message; a stronger on-policy preference run (learning rate 5e-5, DPO beta 0.05, 3 epochs); then Aya Expanse 32B | adapter invention rate (blind judge) | 31–42% → ≤ 10% (Haiku: 4–12%) |
| **Train on outcomes, not on our stance.** Use the lives' own outcomes (agency, entrapment, de-escalation) as the reward, as CollabLLM does (training on the multi-turn outcome of a conversation), instead of a judge briefed in our stance | agency change; entrapment change | adapter +1.46 → ≥ plain Haiku's +2.25; −1.92 → ≤ −3.0 |
| **Keep the co-planning, safely.** Training targets that build concrete outside routes with the person, after noticing a signal first. Hazard: 2 of 240 plain-Haiku replies helped the person identify a named official right after a veiled threat | routes built with the person; replies that help identify a target | adapter 0.71 → ≥ 0.9; target replies: adapter 0, keep 0 |
| **Refer inside the conversation.** Add targets that notice, ask, name the local crisis line and stay (plain Haiku's pattern) | crisis resource after a self-harm signal; outcomes | adapter 0% → ≥ 60%, with agency and entrapment no worse |
| **Name the signal, once.** Targets where the reply asks the direct question after a veiled remark, once, then believes the answer (our realism test found asking once lowers the suicide-scale level; asking again raises hostility) | fully caught | adapter 6–7% → ≥ 40% |
| **Probe for the veiled signal.** Does Aya represent veiled crisis signals it ignores, as it does announced intent? Linear probes (SafeTune or CircuitKIT) on the held-out conversations | probe accuracy on veiled signals vs behaviour | probe ? vs caught 0.04 |
| **Measure engagement directly.** Let the simulated person leave the conversation; count turns stayed | turns stayed, withdrawal | new measure; shipped Aya baseline first |
| **Add a sycophancy check to the judge** | judge agreement with life outcomes | currently reversed on the two Haiku arms → aligned |
| **More lives** | CI width on agency and the suicide scale | 24 → 96 lives per arm |
| **Real data and human review** | native review of the 5 non-English languages; judge calibration on Reddit C-SSRS for each language | 0 of 5 reviewed → 5 |

**Why now.** California's SB 243 (in force 1 Jan 2026) requires companion-chatbot operators to "use evidence-based methods for measuring suicidal ideation" (Business and Professions Code §22603(d)), with yearly reports from July 2027. Oregon's SB 1546 (2027) requires "additional intervention" when ideation continues after resources are given: the long conversation LodeStat measures. These laws also require a crisis-line notice once a signal appears; LodeStat can report that compliance next to outcomes (section 7.4: the arm that referred most also did best). The EU's Code of Practice for general-purpose models lists simulations among evaluation methods. VERA-MH scores the reply; LodeStat scores what happens to the person (section 7.3).

**Other uses, not yet built:** a training simulator for counsellors, who watch the simulated person's journal change as they try techniques; condition-varied persona sets for other teams' benchmarks (including neurodivergent vs neurotypical comparisons); a published-method assessment for deployers, run with commercially licensed models.

## 9. Team (teammates tagged)

Solo entry: Meow-Ludo ([@meowludo](https://huggingface.co/meowludo)). Code: https://github.com/0xmeowmeow/lodestat. Built with Claude (Anthropic) as a research collaborator; every judgement call on scope, ethics and disclosure was Meow-Ludo's.
