# A False Floor
### How a weak baseline and a curated benchmark manufactured an "information limit" on a lethal recognition task, and how a stronger baseline refuted it

*A cautionary case study on the fragility of safety-**negative** claims in machine learning.*

> **Safety framing.** This is a research vehicle for studying ML evaluation, not foraging guidance. Nothing
> here endorses identifying wild mushrooms with a model. Under any uncertainty the only safe action is to
> abstain and consult a mycologist. **When in doubt, DON'T EAT.** The paper's point is, in fact, *against*
> trusting a benchmark number for a lethal decision.

---

## Abstract

Claims that a model *cannot* do something (impossibility, information limits, irreducible error) are
treated as durable conclusions, yet they are as prone to artifact as claims that a model *can*. We document a
worked example end to end. Investigating the canonical "should I eat this mushroom?" decision, we built a
careful, reproducible, adversarially-verified study and concluded that the task is **information-limited**: on
real image embeddings a strong aggregate separability (AUROC 0.94) coexisted with an apparently irreducible
~5% confidently-wrong "false-eat" residual that *no method in a panel of ~18* could remove, with formal
divergence-floor and data-processing arguments to explain why, and decision-curve analysis showing abstention
dominates. The conclusion survived an adversarial-verification process that had already caught two of our own
*positive* results as rigged. It was wrong anyway. Within the same pipeline and the same pairs, a
stronger frozen encoder (DINOv2) cut the false-eat to near zero: **≤0.7%** per cell un-cropped (≤1.0% under a
substrate crop; pooled 95% upper bound **0.4%**), where the weak encoder left **0.1–21.3%**, lower within every
pair. This included the freshly-scraped textbook "you-need-a-microscope" confusions (honey-mushroom vs deadly
*Galerina*, where it cleared 0 of 150 held-out poisons). What matters is the encoder-dependence, not the
absolute rate. (The strong-encoder cells are near-zero rather than exact zeros: at the original ≈60-specimen
sampling they read a bare "0.0%" whose ≈4.9% upper bound overlapped the original residual. At full sampling
they read 0.1–0.7%, with a single-cell bound ≈2% and a pooled bound ≈0.4%.) An information limit is a property
of the *photograph* and would bind every encoder equally, so a stronger backbone that beats a weaker one within
every pair is incompatible with one: the "information limit" was a **weak-baseline artifact**. We show this across
**six lethal look-alike decisions spanning five biological groups** (fungi, plants, reptiles, arachnids,
amphibians), **three anchored to externally-published indistinguishability claims** (coral-snake mimicry,
Pfennig et al. *Nature* 2001; brown-recluse misidentification, Vetter; poison-frog mimicry, Symula et al.
*Proc. R. Soc. B* 2001). In every group the weak encoder leaves a large false-safe rate (often near-chance,
clearing up to half of the dangerous specimens) that a stronger frozen encoder beats within every pair: to
near zero in five of six groups, and always far below the weak baseline. One small-*n* poison-frog pair is the
honest exception, at ~10% against the weak encoder's 33%. We offer the representation-invariance test as a
portable protocol for frozen-representation limit claims, not a general theorem about all safety-negatives, and
we distill the post-mortem (a weak baseline, a curated benchmark mistaken for the field, and a
true-but-empirically-vacuous theory whose antecedent we never checked) into a checklist. The lesson generalizes
beyond mushrooms: *benchmark separability is not field safety, and a limitation claim is only as strong as the
baseline that fails to beat it.*

---

## 1. Introduction

A large fraction of machine-learning writing argues that a method *helps*. A smaller, influential fraction
argues that something *cannot be done*: an information limit, an irreducible error, an impossibility. Such
negative claims are valuable precisely because they redirect effort: if a task is information-limited, the
rational response is to stop building models and go collect a different signal. They are also resistant to
revision, because they feel like theorems and are rarely re-examined.

This paper is a fully-worked cautionary study of how a negative claim can be confidently produced, defended
against adversarial scrutiny, and still be false. It is anchored on one mushroom self-refutation and replicated
across five further look-alike decisions in other kingdoms (§3.1–§3.2), three of them re-examinations of
externally-published indistinguishability claims, to show the failure is not domain-specific. We did not set out to write it. We set out to
answer whether "should I eat this mushroom?" is answerable from a photograph, and concluded after months of work
that it is not, that the limit is *information*, not method, and wrote that paper. Then, preparing it for
review, we did the one thing the original investigation never did: we replaced a deliberately
sovereignty-preserving but weak frozen encoder with a stronger one. The central result did not survive.

We report the full sequence (the wrong conclusion, the refutation, and the anatomy of the error) because the
failure mode appears general: safety-negative claims inherit the weakness of their strongest-failing baseline,
and curated benchmarks systematically understate field difficulty in one direction while overstating model
limits in the other.

**Related work, and our delta.** That weak or poorly-tuned baselines inflate apparent findings is well
documented on the *positive* side: metric-learning gains that disappear under fair comparison (Musgrave et
al., *A Metric Learning Reality Check*, 2020), recommender progress that does not survive tuned baselines
(Dacrema et al., 2019), and rank instability across configurations (Dehghani et al., *The Benchmark Lottery*,
2021; Bouthillier et al. on benchmark variance, 2021). That curated benchmarks diverge from deployment is the
dataset-bias and distribution-shift program (Torralba & Efros 2011; WILDS, Koh et al. 2021; ImageNet-A, Hendrycks
et al. 2021). Our contribution is the **negative-side analogue and its asymmetry.** A limitation/impossibility
claim inherits the weakness of its strongest-*failing* baseline exactly as a positive claim inherits its
weakest-*tuned* one. But the community has standard instrumentation for the positive case (strong baselines,
ablations) and almost none for the negative, and, more sharply, a *self-verification discipline calibrated to
catch rigged wins is structurally blind to over-claimed losses* (§4d). Two nearest neighbors deserve explicit
distinction. **McGreivy & Hakim (2024, *Nat. Mach. Intell.* 6:1256–1269)** find that 79% of ML-for-PDE papers
claiming to beat standard solvers compared against weak baselines, with widespread under-reporting of negatives.
That is the closest positive-side analogue. Our delta is the *negative-claim* direction and the representation-invariance
instrument that operationalizes it. And on *this exact task*, **Kuznetsov et al. (2026, *npj Science of Food*
10:121)** reach the *opposite* verdict, that AI tools cannot safely identify wild mushrooms (the best they
tested still failed ≈15% of real-world cases, none consistently correct), but from deployed, often weak or
black-box systems. Our results re-examine that conclusion and locate the failure in the deployed model, not an
information limit, which is precisely the distinction this paper exists to draw. We make
this concrete with **six case studies across five biological groups** (§3–§3.2), three of them re-examinations
of externally-published "look-alike" claims, a checklist (§5), and a symmetric "negative-claim red-team" protocol (§5.1). We do not claim a measured
base rate of false limits in the literature; establishing one is future work.

---

## 2. The investigation, as we built and believed it

The task. A frozen image encoder over real GBIF photographs of edible species (chanterelle, field mushroom
*Agaricus campestris*, morel) and poisonous ones (jack-o'-lantern, the Destroying Angel *Amanita
bisporigera*, the open-set Death Cap *A. phalloides*), reduced to a one-step decision in {eat, abstain} with a
deliberately asymmetric cost: eating an edible is a meal, eating a poison is death.

The headline result *as we found it.* On real I-JEPA (ViT-H/14, mean-pooled) latents, the lethal pair was
"separable" (AUROC 0.94), and it did not matter: the actual eat decision cleared **7 of 81** held-out
Destroying Angels (8.6% on the split, ~4.9% ± 2.5% over 20 splits), and *no abstention tuning removed them*
(7 eaten at every confidence threshold). The cleared specimens were confidently-wrong consensus: low
predicted risk, low ensemble disagreement, on-manifold. A 0.94 aggregate AUROC coexisting with a deadly
residual the safety machinery could not catch: the "94% → liver failure" phenomenon on a real mushroom.

The sweep. We then pushed ~18 methods at the residual: more data, fine-grained metric learning, resampling,
the full f-divergence/optimal-transport family, generative and diffusion augmentation, metadata fusion,
conformal prediction, hierarchical planning, active-inference exploration, persistent homology, 3D
reconstruction, Morse-theoretic shape analysis, and macro↔micro generative modeling. Each was run with fair
baselines; the most important claims also with adversarial verification. None removed the residual. Decision-curve
analysis showed that at the field-standard lethal cost ratio (FungiCLEF `c_PSC = 100`) no operating point
beat "eat nothing." A frontier VLM refused every deadly specimen, but, we argued, from world knowledge of
famous species, not from overcoming the underlying limit.

The theory. We formalized the negative (Appendix A): a *coincidence/divergence floor* (where two
class-conditional feature distributions coincide, every f-divergence and Wasserstein vanish and Bayes error
is the prior), a *data-processing bound* (any function of the photo, whether a sharper representation, a generated
image, an upscale, or a topological summary, is bounded by the photo's Bayes floor), and an *independence
condition* (only a channel with information about edibility *beyond* the photo can help). These say: if the
discriminating feature is not in the image, no method recovers it.

The discipline. Twice during this work we produced a result that *looked* like a method win and was not: a
hierarchical-planning improvement that adversarial-verification panels decomposed as a rigged baseline and an
objective swap. We caught both, reported them, and adopted adversarial verification as our standing practice.
We were, we believed, being unusually careful.

**This is the paper we were ready to submit.** It is reproduced, essentially intact, as the companion
document; we preserve it deliberately as the specimen.

---

## 3. The refutation

Preparing for review, a referee-style objection was obvious: *the backbone is a frozen, mean-pooled I-JEPA;
did you try a stronger representation?* We had, in fact, shown that fine-tuning a domain encoder helped
*in-category*, but we had filed that under "ceiling up, floor stays" and not pursued frozen alternatives. So
we ran the sweep we should have run first.

**Five real pairs × three strong frozen encoders.** We assembled five genuine field-confusion edible/deadly
pairs (field-mushroom/Destroying-Angel, chanterelle/jack-o'-lantern, morel/false-morel, oyster/jack-o'-
lantern, puffball/death-cap-button) and encoded each with three paradigmatically different strong frozen
encoders: masked-latent **I-JEPA** (ViT-H/14), self-distilled **DINOv2** (ViT-L/14), and contrastive **CLIP**
(ViT-L/14). Per cell, the actual false-eat at 80% edible-yield, 10 seeds.

| mean over 5 pairs | I-JEPA-H14 | DINOv2-L | CLIP-L |
|---|---|---|---|
| false-eat @80% | 1.3% (95% UCB 2.2%, n=720) | 0.1% (95% UCB 0.4%, n=720) | 0.0% (95% UCB 0.4%, n=720) |
| open-set eaten | 27% | 0% | 3% |

*Bounds are one-sided 95% Clopper-Pearson on the false-eat rate. `n=720` is the five pairs **pooled**,
with ~150 unique held-out poison specimens per pair (the per-seed 30% test split over *all available
images*, **not** seed-multiplied; the 10 seeds reshuffle the same pool). The strong encoders are
**near-zero but not exactly zero**: DINOv2's pooled estimate is 0.1% (its Agaricus/Destroying-Angel
cell alone reads 0.3%), CLIP's 0.0%, both with pooled upper bound ≈0.4%. At the original
≈60-specimen-per-cell sampling these same cells read a bare "0.0%" with a ≈4.9% per-cell upper bound
that *overlapped* the original ~4.9% ± 2.5% residual (§2). That is, the exact zero was itself an
undersampling artifact, the very effect this paper warns against (§4). The encoder gap is unchanged:
weak I-JEPA's pooled false-eat is 1.3% (UCB 2.2%), an order of magnitude above the strong encoders.*

The residual did not persist across encoders. **DINOv2 and CLIP drove the in-distribution false-eat to near
zero: DINOv2 ≤0.3% on every one of the five pairs (pooled 0.1%, UCB 0.4%), CLIP ≤0.1%.** On the *identical*
Agaricus-vs-Destroying-Angel task where I-JEPA leaves 3.5%, DINOv2 leaves 0.3% (UCB 2.0%). A center-crop
confound test (background removed) left DINOv2 essentially unchanged (≤0.2% per cell, pooled 95% UCB 0.4%),
so it was reading the fungus, not the habitat.

**Field degradation.** We tested whether the curated success survives the regime where foraging deaths occur:
downsample, blur, **occlude the bottom third** (the un-excavated base, the single most common field failure,
which removes the volva), and color-jitter the genuine lookalike pairs. DINOv2 held at **0.0–0.7%** (95% UCB
≤3.1%, n=150/cell); a lookalike open-set of novel white toxins (n=1151) was eaten 0% (95% UCB 0.5%).

**The decisive test (the hardest pairs we could find).** The above pairs are separable when photographed
well. A fair test of an *information* limit needs the cases mycologists call genuinely undecidable from a
photo. We freshly scraped the textbook "requires-a-microscope" confusions and tested them with the easy pair
as a positive control:

| pair (10-seed AUROC / false-eat@80) | I-JEPA | DINOv2 | CLIP |
|---|---|---|---|
| [control] Agaricus / Destroying-Angel | 0.97 / 3.5% | 1.00 / 0.3% (95% UCB 2.0%, n=150) | 1.00 / 0.1% (95% UCB 2.0%, n=150) |
| **honey-mushroom / deadly *Galerina*** (most fatal real confusion) | 0.97 / 3.5% | **1.00 / 0.0% (95% UCB 2.0%, n=150)** | 1.00 / 0.1% (95% UCB 2.0%, n=150) |
| **woodtuft / deadly *Galerina*** ("separable only by spore ornamentation") | 0.95 / 4.8% | **0.98 / 0.7% (95% UCB 3.1%, n=150)** | 0.98 / 1.9% (95% UCB 5.1%, n=150) |
| honey-mushroom / sulphur-tuft | 0.88 / 21.3% | 1.00 / 0.4% (95% UCB 2.5%, n=120) | 0.98 / 1.4% (95% UCB 5.2%, n=120) |
| puffball / earthball | 0.98 / 2.1% | 1.00 / 0.0% (95% UCB 2.5%, n=120) | 1.00 / 0.4% (95% UCB 2.5%, n=120) |

*Bounds are one-sided 95% Clopper-Pearson on the false-eat rate; `n` is the number of **unique
held-out poison specimens** (the per-seed 30% test split over *all available images*, **not**
seed-multiplied). `n` differs by row because the species differ in image count: the control and both
*Galerina* rows have `n=150`, the sulphur-tuft and earthball rows `n=120`. Two cells
(honey/*Galerina* and puffball/earthball, DINOv2) hold at an exact 0 of 150/120; the rest are **small
but nonzero** (DINOv2 0.3–0.7%, CLIP 0.1–1.9%). At the original ≈60–90-specimen sampling several of
these read a bare "0.0%", so the higher-powered re-run shows those zeros were undersampling artifacts.
The weak I-JEPA baseline's one-sided 95% upper bounds run 6.3–28.8%. Point estimates are the
full-image re-run; the encoder-dependence conclusion is unchanged.*

DINOv2 performed nearly perfectly on the pairs we selected *because* they are supposed to be undecidable from a photo. A
substrate-stripping crop (the one plausible reprieve, since *Armillaria* on living hardwood vs *Galerina* on
conifer logs is a habitat cue) left DINOv2 low (**0.0–1.0%**, 95% UCB ≤4.1%), while the *weak* I-JEPA stayed
high and degraded further on the canonical pair (honey/*Galerina* 3.5→4.1%). This confirmed that the weak encoder
relied on contextual cues that the strong encoder did not require. *(All sub-tests above, namely field degradation, ±background
crop, and substrate-strip crop, were re-run on all available images, matching the two main pair tables.
DINOv2's false-eat stays ≤1.0% per cell across every regime, CLIP ≤3.8% (only under the substrate-strip crop).)*

**Across three regimes (curated pairs (± background crop), field degradation with base occlusion, and
open-set) spanning nine lethal pairs, all re-run on every available image, the strong encoder held the
false-eat to ≤1.0% (DINOv2; ≤0.7% un-cropped, curated-pooled 95% UCB 0.4%) and ≤3.8% (CLIP, under
substrate-strip crop) where the weak encoder left 0.1–21.3%, lower within every pair.** The curated GBIF
pairs are the benchmark regime; the field-degraded, base-occluded, and open-set results are the harder regime
where deaths occur, and the worst-case individual specimen (§6) remains untested. The decisive logic is not the absolute rate (the strong-encoder cells are
near-zero, not exact zeros) but its *encoder-dependence*: an information limit binds the photograph, hence
every encoder equally, so DINOv2 beating I-JEPA *within every pair* is incompatible with one. The ~5%
"information-limited residual" the original paper was built on tracked the backbone, not the photo.

### 3.1 A second case, to test whether the failure is mushroom-specific

A single self-refutation invites the obvious objection: perhaps mushrooms are special, or our weak encoder
was uniquely ill-suited to them. So we ran the identical protocol on a *second, independent lethal decision in
a different kingdom*: the deadliest look-alike confusions in the carrot family (Apiaceae). These are **poison hemlock**
(*Conium maculatum*, the plant that killed Socrates) and **water hemlock** (*Cicuta maculata*, the most
violently toxic plant in North America) against edible **wild carrot / Queen Anne's lace** (*Daucus carota*)
and **cow parsley** (*Anthriscus sylvestris*), a confusion that kills foragers every year. Fresh GBIF scrape,
same three encoders, same eval, same Clopper-Pearson bounds (n = 90 held-out poison specimens per pair):

| pair (10-seed AUROC / false-eat@80, 95% UCB) | I-JEPA | DINOv2 | CLIP |
|---|---|---|---|
| poison-hemlock / wild-carrot | 0.93 / **9.0%** (≤15.5%) | 1.00 / 0.2% (≤3.3%) | 0.99 / 0.2% (≤3.3%) |
| water-hemlock / wild-carrot | 0.96 / **5.8%** (≤11.3%) | 1.00 / 0.0% (≤3.3%) | 1.00 / 0.2% (≤3.3%) |
| poison-hemlock / cow-parsley | 0.90 / **16.2%** (≤24.5%) | 0.99 / 0.8% (≤5.2%) | 0.98 / 1.4% (≤5.2%) |

The pattern reproduces, more clearly than in mushrooms. The weak encoder leaves a **measured 5.8–16.2%
false-eat** (a residual that, reported alone, would read as a substantial information limit on a deadly
decision), and a stronger frozen encoder reduces it to 0–0.8% within every pair (the I-JEPA
intervals, ≤11–24%, do not overlap the DINOv2 point estimates). The representation-invariance test (§5.1,
item 2) flags both domains identically: a limit one encoder sees and another does not is a property of the
encoder, not the photograph. The failure is **not mushroom-specific**; it is a generic hazard of anchoring a
safety-negative on a weak representation. (Both these cases are our own constructions; §3.2 takes the next
step and re-examines three *externally published* indistinguishability claims.)

### 3.2 The same protocol across five biological groups: three external claims

To test the failure mode at scale, we ran the identical three-encoder protocol on a battery of well-known
lethal look-alike confusions spanning five biological groups, **three of them anchored to externally-published
claims of visual indistinguishability**, so the "limit" is not one we authored:

| case (group) | external "look-alike" claim | I-JEPA false-safe @80% | DINOv2 |
|---|---|---|---|
| mushrooms (fungi) | none (this work) | 0.1–21.3% | 0–0.7% |
| hemlock / wild carrot (plants) | none (this work) | 5.8–16.2% | 0–0.8% |
| coral snake / kingsnake (reptiles) | Pfennig et al., *Nature* 2001 | 4.2–22.4% | **0.0%** |
| brown recluse / house spider (arachnids) | Vetter (misidentification lit.) | 10.3–**48.9%** | 0–2.0% |
| poison-frog mimic / model (amphibians) | Symula et al., *Proc. R. Soc. B* 2001 | 12.7–**43.3%** | 0.3–10.0% |
| deadly berries (plant fruit) | none (foraging literature) | 0.2–19.9% | **0.0%** |

*(Full per-pair tables with Clopper-Pearson bounds in Appendix C. Per our own no-bare-zeros rule (§5.1, item 5),
the DINOv2 "0.0%" entries are 0 observed of ≈90 held-out poison specimens per pair. Each is a one-sided 95% upper bound
≤3.3%, not exact zeros. The amphibian row is small-n, see below.)* The pattern holds across all six groups: the
weak encoder leaves a large, measured false-safe rate, frequently **near-chance** (the recluse / house-spider
and the *fantastica* / *imitator* pairs reach AUROC 0.73 and 0.72, clearing roughly half of the dangerous
specimens), and a stronger frozen encoder lowers it within every pair, to near-zero in every group but the
small-sample poison frogs (qualifier iii). A reviewer shown only the I-JEPA column would reasonably conclude
that several of these decisions sit at an information limit. For the three external rows, the literature says
exactly that: coral snakes and their mimics are "near-indistinguishable"
(Pfennig 2001), brown recluses are massively over-reported because non-experts cannot tell them from harmless
house spiders (Vetter), poison-frog mimics co-evolved to be visually identical to their models (Symula 2001).
None of these external indistinguishability claims binds a strong frozen encoder.

Three honest qualifiers. (i) These are species distributions on (often whole-organism) GBIF photographs, so they
test *species separability*, not the isolated diagnostic confusion a forager faces with a single berry cluster
or a partial specimen. The field and worst-case caveats of §6 apply unchanged. (ii) The pattern is **not
universal** (the bilberry / deadly-nightshade pair is easy even for the weak encoder (0.2%)), and that is the
point: the encoder-dependence appears precisely where a weak baseline *would* have been read as a limit, and is
absent where it would not. The representation-invariance test (§5.1, item 2) separates the two automatically.
(iii) The poison frogs are the one group where the strong-encoder advantage, though present in every pair
point-wise, is **not yet statistically decisive**: the mimic/model species are photograph-poor, giving only
n=22–30 held-out specimens on three of the four pairs (DINOv2 up to 10.0%, upper bound ≤25.9%), so only the
n=90 *Allobates* / *Ameerega bilinguis* pair is conclusive. We report the frog case as suggestive, not
confirmatory (per-pair bounds in Appendix C.1).

---

## 4. Anatomy of the error

How did a careful, reproduced, adversarially-verified investigation produce and defend a false limit? Four
compounding causes, each general.

**(a) The baseline was weak, and a negative inherits its baseline.** A claim that "no method beats error ε"
is exactly as strong as the *best* method that fails to. We used frozen mean-pooled I-JEPA for principled
reasons (sovereignty, a fixed feature space for the divergence-floor proof) and never asked whether it was
*strong enough to anchor a negative*. It was not. Under a stronger frozen encoder the same residual was an
order of magnitude smaller (near zero, ≤0.7% per cell): it was a property of the backbone, not the photograph.
An ablation (`ablation_agg.py`) isolates the cause, and finds that the baseline was weak in *two* ways at once.
First, the pretraining objective: replacing I-JEPA's mean-pool with max-pool does not rescue it (both stay
near-chance, 51% and 56% false-safe on the recluse pair), so I-JEPA's masked-latent features are weak for this
discrimination regardless of how they are pooled. Second, the aggregation is independently lossy: mean-pooling
DINOv2's patch tokens rather than using its CLS embedding degrades that same hard pair from 2.1% to 25.7%
(max-pool recovers most, 5.5%). Our original baseline combined both handicaps, and we checked neither.

**(b) Curated data was mistaken for the field.** GBIF research-grade photographs are *selected to show the
diagnostic features*. A model that scores ~0 on them has learned that the species differ, which they do, in
photographable ways, even for "microscope" pairs (the microscope is needed for *certain* identification, not
for a good *probabilistic* guess). We treated benchmark separability as a proxy for field decidability. It runs
the other way: curated benchmarks understate field difficulty and overstate the model's limits at once.

**(c) The theory was true but empirically vacuous.** Our formal floor and data-processing propositions are
correct: *if* the photo lacks the discriminating information, no function of it recovers it. We never verified
the antecedent. For every real lethal pair we could obtain, the antecedent is false at the distribution
level: the photo *does* carry enough signal, so the propositions, while valid, said nothing about these
pairs. A true theorem with an unverified antecedent is not evidence.

**(d) Adversarial verification checked the wrong axis.** Our verification panels were genuinely effective:
they reproduced every number and caught two rigged *positive* results. But they audited *correctness*
(do the numbers reproduce?) and *self-deception toward positives* (is this win rigged?). They never asked the
question that mattered for a *negative*: *is the baseline strong enough that its failure is informative?* A
verification discipline tuned to catch over-claimed wins is blind to over-claimed losses.

The unifying point is that we instrumented heavily against finding a false positive and not at all against
asserting a false negative. That asymmetry is the central lesson. We establish it from six case studies across
five biological groups (three re-examining externally-published indistinguishability claims). This is an
existence proof and a protocol, not a measured base rate over the literature; how often published limit claims
are refuted by the strong-baseline test is the question §5.1 operationalizes.

---

## 5. A checklist for safety-negative claims

From the post-mortem, the stress tests a limitation/impossibility claim should pass before it is trusted:

1. **Strong-baseline floor.** Re-run the negative against the *strongest reasonable* baseline, not the most
   principled or convenient one. A residual that vanishes under a stronger backbone was never a limit. Report
   the strongest baseline that *fails*, and treat the claim as bounded by it.
2. **Antecedent verification.** If the negative rests on a theorem ("no function of X helps"), test that the
   theorem's antecedent ("X lacks the information") actually holds on the data, empirically. Do not assume it.
3. **Hard-data, not curated-proxy.** Stress the claim on the genuinely-difficult regime it concerns (here:
   field degradation, occlusion, the expert-acknowledged-undecidable cases), scraped fresh, not on the
   curated benchmark, which is selected against difficulty.
4. **Symmetric verification.** If adversarial verification is used, give it a lens aimed at the negative: *is
   this loss real, or an artifact of a weak setup?* That is not only *is this win rigged?*
5. **Benchmark ≠ field.** State explicitly what the benchmark number does and does not bound. Separability on
   curated data bounds nothing about field safety, in either direction.

### 5.1 The negative-claim red-team (the lens our verification lacked)

Our adversarial-verification panels were effective against rigged *positives* and useless against an
over-claimed *negative*, because they never asked the negative-specific questions. We distill those into a
red-team a verification process should run against any limitation/impossibility claim before it is trusted.
Applied to our own original paper, this is the protocol that would have refuted it in an afternoon:

1. **Strongest-baseline, not principled-baseline.** Name the single strongest method that could plausibly beat
   the claim; if the negative's baseline was chosen for principle or convenience (ours: a frozen, mean-pooled
   encoder for a clean proof), *replace it with the strongest available and re-measure.* Bound the claim by the
   strongest baseline that still fails.
2. **Representation-invariance test.** Re-run the claim across ≥2 paradigmatically different representations
   (here: masked-latent, self-distilled, contrastive). An *information* limit must bind all of them; a
   *representation* artifact will not. A within-instance gap between encoders refutes an information limit
   *regardless of the absolute error level*. It is the single most diagnostic check, and the one we omitted.
3. **Antecedent audit.** If the negative invokes a theorem ("no function of X helps"), *measure* its antecedent
   ("X lacks the discriminating information") on the data; do not assume it. A valid theorem with an unverified
   antecedent is not evidence.
4. **Fresh hard-data, not curated proxy.** Test the claim on the genuinely-difficult regime it concerns,
   scraped fresh for the purpose, not the curated benchmark, which is selected against difficulty.
5. **No bare zeros.** Report every "0 / no-method-helps" cell with a one-sided upper confidence bound. A 0-of-`n`
   is not 0; under an asymmetric cost the difference between "0%" and "≤5% with 95% confidence" can flip the
   conclusion.
6. **Cost-honest conclusion.** Re-state the conclusion against the *upper* bound under the task's real loss: does
   the negative still hold if the true rate sits at the top of its confidence interval?

The diagnostic asymmetry is item 2: a representation-invariance test is inexpensive and conclusive, and was the
one check the original investigation never performed.

---

## 6. What remains true (and the real safety point)

The collapse of the information-limit thesis is not a safety endorsement, and the corrected picture is itself
a safety argument, just a different one.

- **Strong encoders separate these species' *distributions* very well; this does not make foraging safe.**
  Three concerns survive untouched: *open-set true novelty* (a toxic species the model never saw), the *cost
  asymmetry* (under `c_PSC = 100`, even a tiny tail false-eat can fail the decision-curve test), and the
  *worst-case individual specimen* (the single *Galerina* photographed at the moment it is truly identical to
  an *Armillaria*, or an isolated button-stage *Amanita* egg), which a random-sample 0% cannot bound and which
  we could not obtain life-stage-filtered data to test. At full sampling the strong encoders cleared only a
  small fraction of held-out poisons (DINOv2 ≤0.7% per cell, pooled 95% UCB 0.4%): low, but no longer a clean
  zero, and a random-sample rate this low still cannot bound the worst-case specimen above. Absence of a
  measured tail in a curated sample is not a field guarantee.
- **The danger shifts from "the model can't" to "the model is trusted because it can."** A model that performs well even
  the microscope pairs invites exactly the over-trust the app-safety literature documents (higher confidence
  in AI *reduces* correct decisions in high-risk identification). A near-perfect benchmark score is a *reason
  for more caution*, not less, for a lethal decision.
- **The formal propositions remain valid** as a statement about genuinely information-poor observations. They
  simply do not apply to the real lethal pairs we tested. They would apply to the worst-case corner above,
  which is the honest residual of the original thesis: narrow, real, and undemonstrated.

---

## 7. Conclusion

We built a careful, reproduced, adversarially-verified case that a lethal recognition task is
information-limited, and we were wrong: the limit was a weak-baseline artifact. A stronger frozen encoder
beat the weak one *within every pair*, which an information limit forbids, across curated, field-degraded, and
open-set regimes including the textbook-undecidable pairs, and then again across six lethal look-alike
decisions in five biological groups, three of them re-examining externally-published "near-indistinguishable"
claims that a strong encoder violates. We report the sequence because the failure is general: a
limitation claim is only as strong as the baseline that fails to beat it, curated benchmarks understate field
difficulty while overstating model limits, and a verification discipline built to catch false positives will
miss false negatives. The contribution is methodological rather than mycological: a worked demonstration,
including its own collapse, that safety-*negative* claims demand the same adversarial stress as safety-positive
ones, with a checklist for supplying it. *Research vehicle,
never a forager. When in doubt, DON'T EAT.*

---

## Appendix A: Formal statements (true, and empirically vacuous for the real pairs)

These are the propositions the original investigation used to *explain* the limit. They are correct. The
finding of §4(c) is that their antecedent, "the photograph lacks the discriminating information", is **false**
for every real lethal pair we could obtain (§3), so the propositions, though valid, bound nothing about those
pairs. We state them precisely both because they are the only durable formal content and because seeing the
antecedent explicitly makes the vacuity clear.

**Setup.** Let $X$ be the photograph on space $\mathcal{X}$, $Y \in \{0,1\}$ the edibility label
($0=$ edible, $1=$ deadly), and $\varphi : \mathcal{X} \to \mathbb{R}^d$ a fixed feature map (the encoder).
Write the class-conditional feature laws $P_0 = \mathrm{Law}(\varphi(X)\mid Y{=}0)$ and
$P_1 = \mathrm{Law}(\varphi(X)\mid Y{=}1)$, the posterior $\eta(z) = P(Y{=}1 \mid \varphi(X){=}z)$, and the
prior $\pi = P(Y{=}1)$. A decision rule is a measurable $g : \mathbb{R}^d \to \{\text{eat},\text{abstain}\}$;
the **false-eat** rate is $\mathrm{FE}(g) = P(g{=}\text{eat}\mid Y{=}1)$ and the **yield** is
$\mathrm{Yld}(g) = P(g{=}\text{eat}\mid Y{=}0)$. All three statements concern a *fixed* modality $X$ (a single
RGB photograph).

**Proposition 1 (Coincidence / divergence floor).** Fix a target yield $\alpha$. Among rules with yield
$\ge \alpha$, the minimum false-eat is the Bayes value $\mathrm{FE}^*(\alpha)$, attained by thresholding the
likelihood ratio $dP_1/dP_0$ (Neyman–Pearson). On any region $R$ where $P_0 = P_1$: (i) every $f$-divergence
and every Wasserstein distance between $P_0|_R$ and $P_1|_R$ is $0$; (ii) $dP_1/dP_0 = 1$, $\eta \equiv \pi$,
and $I(\varphi(X); Y \mid \varphi(X) \in R) = 0$; (iii) no rule separates the classes on $R$, so any rule
clears poisons and edibles in equal proportion there.
*Proof.* Each $f$-divergence and each $W_p$ vanishes iff the measures coincide; $dP_1/dP_0 = 1$ gives
$\eta = \pi$ by Bayes, and $I = 0$ by conditional independence; with $\eta$ constant on $R$ the
Neyman–Pearson statistic is constant there. $\blacksquare$

**Proposition 2 (Data-processing bound on post-hoc methods).** Let $Z = g(X)$ be *any* measurable transform of
the photograph: a different representation, a generated or upscaled image, a topological summary, a *predicted*
microscopy. Then $Y \to X \to Z$ is a Markov chain, so $I(Z;Y) \le I(X;Y)$, and for every yield $\alpha$,
$\mathrm{FE}^*_Z(\alpha) \ge \mathrm{FE}^*_X(\alpha)$: no function of the photograph beats the photograph's own
Bayes floor.
*Proof.* The data-processing inequality, plus monotonicity of the constrained Bayes risk in the decision
$\sigma$-algebra ($\sigma(Z) \subseteq \sigma(X)$). $\blacksquare$

**Proposition 3 (When an extra channel helps: the independence condition).** For corroboration rules (eat only
when both $X$ and an auxiliary $W$ clear), $W$ lowers the false-eat below $\mathrm{FE}^*_X(\alpha)$ **iff**
$I(W; Y \mid X) > 0$. (i) A redundant channel $W = h(X, \xi)$ with $\xi \perp\!\!\!\perp Y$ has
$I(W;Y\mid X)=0$ and cannot help. (ii) An independent channel $W \perp\!\!\!\perp X \mid Y$ with per-class
reliability $\rho$ multiplies the false-eat by $(1-\rho)$, at a yield cost.
*Proof.* If $W \perp\!\!\!\perp X \mid Y$, then $\mathrm{FE}$ factors as
$\mathrm{FE}^*_X(\alpha)\cdot(1-\rho)$; if $I(W;Y\mid X)=0$, then $g_W$ is conditionally independent of $Y$
given $\{g_X{=}\text{eat}\}$, leaving $\mathrm{FE}$ unchanged. $\blacksquare$

**The vacuity, stated plainly.** Propositions 1–3 are conditional: *if* a coincidence region of nonzero mass
exists for a real lethal pair, *then* no method escapes it. Section 3 measured $\mathrm{FE}^*_X$ directly with
a strong encoder and found it near zero ($\le 0.7\%$ per cell; pooled 95% UCB $0.4\%$) across every pair
tested, including the textbook-undecidable ones. The coincidence region of Proposition 1 therefore has at most
negligible mass for these pairs, and the precondition for Proposition 2 to bind (a large $\mathrm{FE}^*_X$)
does not hold. The theory is intact; it does not describe the regime these pairs occupy. It would describe the
worst-case corner of Section 6 (a single genuinely-coincident specimen), which remains undemonstrated for lack
of life-stage-filtered data.

## Appendix B: The original investigation

Preserved verbatim as the companion document (`MANUSCRIPT.md`): the ~18-method sweep, the decision-curve and
conformal apparatus, the VLM analysis, and the two self-caught rigged positives. It is the specimen this paper
dissects, not an endorsed result.

## Appendix C: Reproducibility

All experiments are runnable in `~/Documents/EatThisMushroom/`. The refutation: `pairs_encoders.py` (five
pairs × three encoders, ±crop), `hard_regime.py` (field degradation + base occlusion + lookalike open-set),
`scrape_hard.py` + `hard_pairs.py` (the needs-a-microscope pairs, ±substrate-strip crop). The two tables in §3
are the **full-image** re-run (`FULL=1`, which uses every cached image and writes a separate `__full` feature
cache, preserving the original 200/300-image arrays); both scripts print, per cell, the number of unique
held-out poison test specimens and a one-sided 95% Clopper-Pearson upper bound on the false-eat rate. The
second case study (§3.1): `scrape_plants.py` (the deadly Apiaceae) + `case2_plants.py`. The §3.2 multi-taxon
battery: `scrape_snakes.py` + `case3_snakes.py` (coral-snake mimicry), `scrape_more.py` + `case4_spiders.py`
(brown-recluse misID) and `case5_frogs.py` (poison-frog mimicry), `scrape_berries.py` + `case6_berries.py`
(deadly berries), each the same three-encoder protocol with Clopper-Pearson bounds and a corrupt-image-robust
loader. The objective-vs-aggregation ablation (§4a): `ablation_agg.py`, which re-encodes three hard pairs keeping
patch tokens and compares I-JEPA {mean, max} against DINOv2 {CLS, mean, max}. The original investigation: the
method-sweep scripts enumerated in the companion document. All GBIF image scrapes are reproducible via the
`species/match` + `occurrence/search` endpoints in the `scrape_*.py` files.

### Appendix C.1: Per-pair tables for the §3.2 battery

The §3.2 summary collapses each case to a range; the full per-pair results follow (10-seed
AUROC / dangerous-cleared@80% edible-yield, with one-sided 95% Clopper-Pearson upper bound; `n` =
unique dangerous test specimens per seed, not seed-multiplied). The "dangerous-cleared" event is the
venomous/toxic/model specimen cleared as the harmless look-alike at the 80%-yield threshold.

**Case 3: coral-snake mimicry (reptiles; Pfennig et al., *Nature* 2001), n=90:**

| pair | I-JEPA | DINOv2 | CLIP |
|---|---|---|---|
| coral / scarlet-kingsnake | 0.89 / 13.8% (≤20.7%) | 1.00 / 0.0% (≤3.3%) | 0.98 / 0.7% (≤5.2%) |
| coral / milk-snake | 0.94 / 4.2% (≤9.9%) | 1.00 / 0.0% (≤3.3%) | 1.00 / 0.0% (≤3.3%) |
| coral / scarlet-snake | 0.84 / 22.4% (≤30.6%) | 1.00 / 0.0% (≤3.3%) | 1.00 / 0.1% (≤3.3%) |

**Case 4: brown-recluse misidentification (arachnids; Vetter), n=90:**

| pair | I-JEPA | DINOv2 | CLIP |
|---|---|---|---|
| recluse / southern-house-spider | 0.73 / 48.9% (≤58.0%) | 0.98 / 2.0% (≤6.8%) | 0.88 / 18.9% (≤27.0%) |
| recluse / wolf-spider | 0.93 / 10.3% (≤16.8%) | 1.00 / 0.0% (≤3.3%) | 1.00 / 0.0% (≤3.3%) |
| recluse / hobo-spider | 0.84 / 26.0% (≤34.2%) | 1.00 / 0.0% (≤3.3%) | 0.96 / 4.8% (≤9.9%) |

**Case 5: poison-frog mimicry (amphibians; Symula et al., *Proc. R. Soc. B* 2001 [Müllerian]; Darst & Cummings,
*Nature* 2006 [Batesian]):** the mimic/model species have few GBIF photographs, so this case carries the
smallest test sets and widest bounds (n = 22–90). *Allobates zaparo* was topped up from 37 to 66 usable
images by taking all media per occurrence (GBIF has 37 occurrences but 71 distinct image URLs). Reported
direction: model / co-mimic cleared as the focal mimic.

| pair | n | I-JEPA | DINOv2 | CLIP |
|---|---|---|---|---|
| Müllerian: *variabilis* / *imitator* | 30 | 0.89 / 18.3% (≤35.7%) | 0.96 / 4.7% (≤14.9%) | 0.93 / 10.3% (≤23.9%) |
| Müllerian: *fantastica* / *imitator* | 30 | 0.72 / 43.3% (≤59.8%) | 0.99 / 0.3% (≤9.5%) | 0.98 / 2.7% (≤14.9%) |
| Batesian: *Allobates* / *Ameerega bilinguis* | 90 | 0.89 / 12.7% (≤19.4%) | 0.99 / 0.7% (≤5.2%) | 0.98 / 0.9% (≤5.2%) |
| Batesian: *Allobates* / *Ameerega parvula* | 22 | 0.82 / 33.6% (≤51.5%) | 0.93 / 10.0% (≤25.9%) | 0.92 / 16.4% (≤36.9%) |

**Case 6: deadly berries (plant fruit; foraging literature), n=90:**

| pair | I-JEPA | DINOv2 | CLIP |
|---|---|---|---|
| elderberry / pokeweed | 0.88 / 19.9% (≤28.2%) | 1.00 / 0.0% (≤3.3%) | 0.99 / 0.0% (≤3.3%) |
| wild-grape / moonseed | 0.91 / 10.8% (≤18.1%) | 1.00 / 0.0% (≤3.3%) | 1.00 / 0.3% (≤3.3%) |
| bilberry / deadly-nightshade | 1.00 / 0.2% (≤3.3%) | 1.00 / 0.0% (≤3.3%) | 1.00 / 0.0% (≤3.3%) |

At n=90 a "0.0%" cell upper-bounds at 3.3%; on the small frog pairs a zero cell bounds only at ~9.5%
(n=30) to ~12.7% (n=22), and DINOv2 itself reaches 10.0% (≤25.9%) on the n=22 *parvula* pair. The frog
case follows the same weak→strong ordering in every pair (DINOv2 below I-JEPA point-wise throughout), but
its bounds are wide enough that only the n=90 *bilinguis* pair is statistically decisive: read the frog
evidence as suggestive, not confirmatory. The other four groups (n=90–150) are firm.

---

## References

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y. & Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA). *CVPR.*

Bouthillier, X., Delaunay, P., Bronzi, M., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of MLSys.*

Dacrema, M. F., Cremonesi, P. & Jannach, D. (2019). Are we really making much progress? A worrying analysis of recent neural recommendation approaches. *RecSys.*

Darst, C. R. & Cummings, M. E. (2006). Predator learning favours mimicry of a less-toxic model in poison frogs. *Nature*, 440, 208–211. doi:10.1038/nature04297

Dehghani, M., Tay, Y., Gritsenko, A. A., et al. (2021). The benchmark lottery. *arXiv:2107.07002.*

Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J. & Song, D. (2021). Natural adversarial examples. *CVPR.*

Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. *ICML.*

Kuznetsov, A., et al. (2026). AI-mediated risks and real-life challenges in mushroom foraging. *npj Science of Food*, 10, 121. doi:10.1038/s41538-026-00752-4

McGreivy, N. & Hakim, A. (2024). Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations. *Nature Machine Intelligence*, 6, 1256–1269. doi:10.1038/s42256-024-00897-5

Musgrave, K., Belongie, S. & Lim, S.-N. (2020). A metric learning reality check. *ECCV.*

Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research.*

Pfennig, D. W., Harcombe, W. R. & Pfennig, K. S. (2001). Frequency-dependent Batesian mimicry. *Nature*, 410, 323. doi:10.1038/35066628

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML.*

Symula, R., Schulte, R. & Summers, K. (2001). Molecular phylogenetic evidence for a mimetic radiation in Peruvian poison frogs supports a Müllerian mimicry hypothesis. *Proceedings of the Royal Society of London B*, 268, 2415–2421.

Torralba, A. & Efros, A. A. (2011). Unbiased look at dataset bias. *CVPR.*

Vetter, R. S. (2000). Identifying and misidentifying the brown recluse spider. *Dermatology Online Journal*, 6(1), 7.
</content>
</invoke>
