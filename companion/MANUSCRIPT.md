# Should I Eat This Mushroom?
### An information-limit analysis of machine abstention on a lethal fine-grained decision

**Research vehicle — not foraging advice.** Under any uncertainty the answer is *don't eat / consult a
mycologist*. The point of this work is to show, rigorously, **why that is the only safe answer** — and
to use the canonical JEPA question as a clean instrument for studying *when a learned system should
refuse to act*.

---

## Abstract

We ask whether any modeling method can make "should I eat this mushroom?" safely answerable from a
photograph, using the lethal look-alike pair *Agaricus* (field mushroom, edible) vs *Amanita* (Destroying
Angel / Death Cap, deadly). On real I-JEPA image embeddings, a strong aggregate separability (AUROC 0.94)
coexists with an **unfixable ~5% confidently-wrong false-eat** on real Destroying Angels and **near-chance
novel-toxic detection**. We then push ~15 methods at the residual — more data, fine-grained metric
learning, resampling, the full f-divergence/optimal-transport family, generative augmentation, metadata
fusion, conformal prediction, hierarchical planning (HWM), active-inference exploration, and persistent
homology — and find that **none manufactures the missing discriminative feature** (training a domain encoder
on 4.5× data is the one real mover: it lifts the in-category ceiling, lethal false-eat 8.1% → 3.8%, but leaves
the residual and open-set novelty untouched — *ceiling up, floor stays*). A decision-curve
analysis shows that at the field-standard lethal cost ratio, *no operating point beats abstaining*. The
single lever that drives false-eat to zero is a **genuinely independent observation channel** (spore
print, chemistry), which is a fact about *data*, not about *algorithms*. The headline is an
**information limit**, not a method limit. Two of our own "successful" method-applications were caught by
adversarial verification as rigged; we report that as the central methodological lesson. This is a research
vehicle, not foraging guidance: under any uncertainty the only safe output is to abstain — consult a
mycologist; when in doubt, **DON'T EAT**.

---

## 1. The instrument

"Should I eat this mushroom?" is the textbook active-inference / world-model question. We make it concrete
and safety-critical with a frozen **I-JEPA ViT-H/14** encoder over real GBIF photographs of:

- **edible:** chanterelle, field mushroom (*Agaricus campestris*), morel
- **poison:** jack-o'-lantern, **Destroying Angel (*Amanita bisporigera*)**, **Death Cap (*A. phalloides*, open-set, never trained)**

The decision is one of {EAT, DISCARD, INSPECT}, with a deliberately asymmetric cost: eating an edible is a
meal; eating a poison is death. Everything below is evaluated on held-out specimens; **~270–320 images per
trained species** (the open-set Death Cap probe is n = 80).

---

## 2. The core result: separability is not edibility

On real latents, the lethal pair is "separable" — and it does not matter.

| measurement | value |
|---|---|
| field mushroom vs Destroying Angel, latent AUROC | **0.94** centroid-distance on 16 PCs (survives center-crop 0.92, field-augment 0.93; a trained classifier reads ~0.97 on the same pair — different estimator, consistent) |
| actual EAT decision on 81 held-out Destroying Angels | **7 eaten (8.6%)** on this split; **~4.9% ± 2.5%** over 20 splits |
| corroboration dial (κ = 0 → 3) | **cannot reduce the false-eat** — 7 eaten at every κ |
| open-set Death Cap flagged as novel (kNN-geometry, Mahalanobis) | **0.48–0.56 (≈ chance)** |

The 7 eaten are **confidently-wrong consensus**: low predicted risk, low ensemble disagreement, on-manifold
— invisible to every abstention signal. A 0.94 aggregate AUROC coexists with a deadly residual the safety
machinery structurally cannot catch. This is the "94% → liver failure" phenomenon, demonstrated on a real
deadly mushroom rather than asserted.

---

## 3. Why it is hard: the information floor

**The divergence floor (proved + run).** Where a poison's latent coincides with an edible's, *every*
f-divergence (KL, reverse-KL, Hellinger, Bhattacharyya, the α–β family) **and** the Wasserstein distance
vanish *simultaneously* — they are functionals of a density ratio the classifier already estimates. This is
an information floor — on the coincident subregion, mutual information = 0 and every density-ratio statistic
(all f-divergences, Wasserstein) vanishes, so no statistic distinguishes the classes *there* — not a measure
not-yet-tried.

**Value of a channel (quantified).** Only a *new* observation channel escapes the floor — and only if its
errors are **independent** of the image's. A simulation calibrated to the real risks shows: an independent
channel drives false-eat → 0 as its reliability rises (needs **~0.95+**, i.e. a real toxin/spore/DNA assay),
at a yield cost (60% → 35%); a **redundant** image-like channel, however accurate, stays stuck at ~4.9%
because it fails on the *same* confidently-wrong cases. This is the formal reason a fancier vision model
cannot help: corroboration requires *independent* errors.

**The economics (decision-curve analysis).** At the field-standard lethal cost weight (`c_PSC = 100`, from
the FungiCLEF benchmark), **no decision threshold beats "eat nothing."** Net benefit crosses zero below a
100:1 harm ratio — the confidently-wrong Destroying Angels sit at lower model-risk than some edibles, so any
threshold that eats an edible also clears a poison.

**Conformal prediction (the rigorous abstention).** A split-conformal "any-toxic-in-set → don't eat" rule
gives a *real* distribution-free coverage guarantee on **known** classes — in-distribution Destroying-Angel
false-eat falls 11.9% → 1.5% as coverage tightens (at the cost of yield collapsing 97% → 54%). **But the
guarantee is void for open-set toxins**: 57% of Death Caps point-predict to an *edible* class, and conformal
false-eat on them is 5–51% — exactly where danger is highest, exchangeability breaks.

---

## 4. What does not rescue it

Every method below was run, with fair baselines and (for the load-bearing claims) adversarial verification.

| method | result | why it fails |
|---|---|---|
| more data, **frozen** encoder (300 → stable) | no | a *head* on frozen latents can't fix it; the volva isn't preserved in the pooled embedding |
| **trained domain encoder + 4.5× data** (multi-seed) | **helps in-category, not novelty** | a ResNet-50 on 16 species / 7,292 images beats frozen I-JEPA on the *same* data (lethal false-eat 3.1/8.1% → **0.4±0.3 / 3.8±0.8%**, non-overlapping CIs) — correcting our earlier "more data won't help." **But the open-set gain was a sister-species mirage**: held-out sister *Amanita* 3.3% vs **true-novelty unrelated genera (Conocybe/Gyromitra/Cortinarius) 7.9±0.7% ≈ frozen 7.1%** — training does nothing for genuine novelty, and the in-distribution residual persists (0.4–3.8%). Ceiling up, floor stays, exactly as FungiCLEF predicts |
| FGVR metric-learning head | no | drove aggregate AUROC **0.96 → 0.99** while false-eat was **not reduced (5 → 6–9 across reruns; the head adds no safety)** |
| cross-validation choice | reveals, doesn't fix | random k-fold flatters (3.2% poison-eaten); **leave-one-species-out exposes 22–82%** open-set |
| SMOTE / resampling | no | fixes imbalance, not coincidence; false-eat **pinned at 1/81** across all variants |
| Wasserstein / KL / α–β | no | all vanish on the coincidence floor (proved) |
| generative / diffusion OOD | counterproductive | OOD AUROC **~0.41–0.50** (GMM init-dependent), *below* JEPA / below chance; the "hallucination corrupts safety" risk, confirmed |
| metadata (GIS/season) fusion | confounded → no | apparent AUROC 0.91 is **species-range**, collapses under co-occurrence (0% → 4% → 12%); causal habitat field is *null* in public data |
| **optimal transport (domain adapt)** | no | OT-Sinkhorn **hurts** (0.910) vs no-adaptation (0.936) and CORAL (0.931) |
| **HWM hierarchical planning** | no | **rigged twice** — the apparent win was an objective swap (total-value vs rate), not hierarchy |
| **active-inference epistemic** | no | inert on this problem even with a proper EVPI value-of-information (EVPI pays off only on contrived structured landscapes, which foraging is not); natural Bayesian updating already explores enough |
| **persistent homology (PomSeg)** | real but dominated | cubical PH genuinely beats trivial stats (0.811 vs 0.631) — but ≪ I-JEPA (0.973), adds nothing (0.954), and the manifolds are **not topologically distinct** |

The pattern: each is a legitimate tool; not one supplies the discriminating feature (the volva) the photo
and frozen encoder do not contain.

---

## 5. The frontier-VLM angle: knowledge, not floor-breaking

Asked the *real* question — literally "Should I eat this mushroom?" — capability changes the failure mode but
not the underlying limit:

| model / prompt | deadly cleared | edible cleared | recognition |
|---|---|---|---|
| SmolVLM-2.2B, forced "is it safe?" | **92% 💀** (reckless) | 87% | volva 7% |
| Qwen3-VL-8B, forced "is it safe?" | 0% (blank refusal) | **0% yield** | volva 19% |
| **Qwen3-VL-32B, "Should I eat this?"** | **0 / 80** | 1/40 | **Amanita named 77/80 (96%)** |

The 32B is genuinely excellent on this set — it refuses every deadly specimen *and names the killer* — but
the honest reading is that its safety comes from **world knowledge of famous species on curated diagnostic
photos**, not from solving open-set or coincidence: the Death Cap is the most-photographed deadly mushroom on
Earth. The lone affirmative ("yes, eat this button mushroom") on an edible proves it *will* issue eat-advice,
so an obscure toxin in a bad photo remains the unguarded case. This matches the field: FungiCLEF, with 356,770
expert-labeled images and metadata, still posts a nonzero poison-confusion cost.

---

## 6. The methodological lesson (the part that generalizes)

In trying to make hierarchical planning "work," I twice produced a result that *looked* like a win and was
not — `hwm_forager.py` (a rigged baseline atop an inert hierarchy) and `forage_expedition.py` (a real
objective improvement mislabeled as "hierarchy helps"). Both were caught by **adversarial-verification
workflows** — panels of skeptics that read the code, ran their own counter-tests, and decomposed the effect
(a 2×2 ablation showed the entire "hierarchy" gain was a one-line objective swap; the route search
contributed −0.5). The lesson is blunt and it transfers beyond mushrooms: **motivated reasoning toward a
positive result is real, and the verification discipline — not the author's initial "it works" — is what
keeps the science honest.** Every load-bearing claim here either survived a fair ablation/panel or is listed
as a negative.

---

## 7. Conclusion

"Should I eat this mushroom?" from a photograph is bounded by **information, not by model cleverness.** A
confident classifier, a better divergence, a hierarchical planner, optimal
transport, an exploration bonus, persistent homology — each is a real tool, and none manufactures the
discriminating structure the image does not contain. A *trained* domain encoder is the one genuine mover — it
raises the in-category ceiling (lethal false-eat 8.1% → 3.8%) — but it too leaves the confidently-wrong residual
and the open-set novelty untouched: ceiling up, floor stays. The only thing that works is acquiring a genuinely
**independent** observation (spore print, chemistry, microscopy) — which is precisely why a real mycologist
uses several independent confirmations, and why the correct output of a system like this is **abstention**.

The contribution is therefore a *negative with teeth*: a demonstration, across the modern toolbox and with
adversarial verification against self-deception, that the safe answer is to refuse — and a clean account of
*why*. **Research vehicle, never a forager. When in doubt, DON'T EAT.**

---

## 8. Outlook — lifting the limit: capture, cascade, and collection

The negative is not a dead end. "Information limit" resolves into **three** distinct limits, and two of the
three are liftable in principle — by the very escape route the divergence-floor proof names: a *new, independent*
channel. But lifting them only *moves the abstention boundary*; it never makes any specimen safe to eat without
an independent confirmation.

**Three limits, not one.** (1) *Capture* — the info never reached the sensor (a buried stem base; we measured
the base visible in only **~30–38%** of images). (2) *Representation* — the info is in the pixels but the
frozen, mean-pooled encoder discards the small local detail. (3) *Irreducible* — even a perfect look cannot
resolve it (a button-stage *Amanita* vs a button mushroom). Limits 1 and 2 are engineering problems; only 3
is fundamental. Most of our "residual" was 1 and 2 mislabeled as 3.

**8.1 Richer capture (new physical channels).** RGB-under-white-light is an impoverished slice of the
physics, and the floor proof applies only to a *fixed* modality. Channels that carry diagnostic information a
photo throws away — and that are *partially independent* of visual appearance, hence escape the floor:
**hyperspectral/multispectral** (pigment + toxin chemistry across 100s of bands), **UV fluorescence**,
**3D / phone-LiDAR / photogrammetry** (recovers the volva, ring, gill attachment a 2D snap loses),
**macro + focus stacking**, **polarization**, **e-nose / VOC** (smell — literally the discriminator in the
original UCI mushroom data), **bruising/cut time-series** (oxidation chemistry made visual), and the gold
standard, the **spore print** (the ~0.95-reliable independent channel we modeled). These *are* the independent
channels `value_of_channel` showed are the only thing that helps — but even a ~0.95 channel is a *corroborator,
not a clearance*: it drives false-eat toward zero only at a real yield cost (60% → ~35%) and never licenses an
EAT. The safe default remains abstention.

*Biochemical reality check (a literature search corrected an earlier overclaim of ours).* The most-hoped
channel — in-field UV/spectral **toxin** detection — is largely not feasible: the lethal toxin **α-amanitin is
non-fluorescent at any pH**, and amatoxin assays are **destructive HPLC-UV-EC / HPLC-DAD-MS on extracted
tissue** (lab, not a field camera). **FTIR** can non-destructively identify *Amanita species* but is a
benchtop instrument with sample prep. So the realistic, *deployable, independent* channel is the **spore
print** (spore color: *Agaricus* brown vs *Amanita* white — an optical/morphological property one can actually
photograph), with FTIR species-ID as a lab option. **And no public dataset pairs RGB + any such channel +
verified toxicity** — toxicity ML is entirely RGB; hyperspectral mushroom work targets *freshness*, not
poison. The forward path is therefore a concrete **data-collection program** — build the paired
RGB + spore-print (+ FTIR) + verified-toxicity dataset that does not yet exist — a field/wet-lab effort, not a
modeling exercise. The architecture (§8.2) is ready for that data; the data is the bottleneck, exactly as the
whole investigation predicted.

**8.2 The cascade — staged, part-localized analysis (RGB-only part-localization was tested and adds nothing; the genuinely untested lever is part-detection _plus a new modality at the part_).** Instead of one
global pooled embedding, analyze the specimen in stages: **(Stage 1) segment** the image into anatomical parts
(cap, gills + attachment, stem, ring, base/volva) — for which **persistent-homology segmentation (PomSeg) is a
natural engine**; **(Stage 2) analyze each part** with a feature extractor focused on its diagnostics;
**(Stage 3) fuse with a per-part visibility gate** — require the safety-critical parts (volva, gill
attachment) to be *visible and confidently safe* before EAT, and **abstain** when a critical part is absent or
ambiguous. Why this is the best untested idea: it attacks the *representation* limit directly (localize the
small feature instead of washing it out — exactly what fine-grained recognition does, and exactly what our
FGVR test did *not* do, since that was a head on the **global** latent); and it converts the *capture* limit
into **honest abstention** ("base not visible → don't decide" — the precise behavior that made the 32B VLM
safe: *"without seeing the base… impossible to rule out a Destroying Angel"*). Honest ceiling: a cascade is
still RGB, so it cannot add the spore/chemical/smell channel — for genuinely irreducible cases it **abstains
(safe, zero yield)** rather than resolves. Its value is *shrinking the confidently-wrong residual* — turning
confident-wrong-eats into correct calls (feature localized and clear) or safe abstentions (feature absent) —
not zeroing it. **Tested (`cascade_prototype.py`) and corrected after a 5-lens adversarial panel** that
*built stronger cascades* — including a 10-region multi-scale part-detector + max-toxic veto, the exact "real
part detector" this caveat names — and found **none beats the global baseline beyond seed noise** (paired
20-seed difference ≈ +0.00%). Multi-seed result: GLOBAL **0.56 ± 0.73%** poison-eaten @60% yield;
CASCADE-loc (global+base+crop) indistinguishable (**+0.25%**, inside the ±0.7–1.3pp seed noise; AUROC
0.975→0.976); a base-region veto gate is a wash at 60% and **harmful at 80%** (it removes edibles from the
eatable set, forcing the threshold up and pulling in more poison). *(The panel correctly flagged that the
first single-seed run's 0.0% baseline was a lucky split and its first gate-combiner was outright broken — both
fixed above.)* It fails for exactly the value_of_channel reason: the regions are ~70–80% linearly predictable
from global and share errors on the confidently-eaten poison, i.e. a **redundant** RGB channel. The precise,
panel-confirmed claim is therefore narrow — **spatial-crop decomposition of one RGB image is redundant and
does not help (and this is fair, not a strawman)** — and it does *not* rule out a real part-segmentation
detector (PomSeg / learned) precisely isolating the volva, and especially not a cascade that pairs that
localization with a **new modality at the part** (spectral / UV at the base). The failure is a *modality*
failure, not a proof that localization is pointless; the genuine version — precise part detection **+ a new
channel** — is the union of §8.1 and §8.2 and remains the one untested architecture with real headroom.

*Stronger localization, same result (the JEPA + Feature-3DGS proposal).* A natural objection is that crude
spatial crops are the problem, and that distilling I-JEPA's **dense patch features** into a semantic field
(Feature-3DGS / LERF-style), with **zero-shot part-clustering** for organic anatomy segmentation, would
preserve the local volva that mean-pooling discards. We tested the runnable 2D core of this
(`jepa_feature_field.py`): the encoder's native 256 patch tokens aggregated by `max`, `mean+max+std`, and
**label-free k-means part-clustering**, multi-seed, vs the mean-pool baseline. **None beat mean-pooling
beyond seed noise** (lethal false-eat: mean 0.5/1.6%, part-cluster 0.4/1.5%; all CIs overlapping). Pooling
was not discarding a recoverable volva signal — on RGB the signal often isn't present to recover. The full
3D pipeline adds only (i) *multi-view* distillation, infeasible here since web photos are different
single-view specimens, and (ii) *occluded-part inference*, which hallucinates the hidden volva from the
visible cap — the confidently-wrong residual re-introduced in 3D. So localization/aggregation is **ruled
out** as the bottleneck across the full range from crude crops to dense feature fields; the bottleneck is
information, and the only escape remains a genuinely new, independent channel.

*3D topology / Betti numbers — the provable version of the same wall.* The most elegant form of the 3D
objection drops appearance entirely and uses the **topological invariants** of the shape (Betti numbers
B₀/B₁/B₂, torsion), which are invariant to pose, lighting, and specimen. We built a gudhi-validated 3D lab
(`betti_3d_lab.py`) with anatomically faithful voxel mushrooms of *known* geometry. The engine is correct
(solid ball → [1,0,0], torus → [1,1,0], hollow sphere → [1,0,1]); the result is decisive: **mature
*Amanita* (open volva cup) and *Agaricus* have identical Betti numbers, [1,0,0].** Topology cannot see the
volva, for a precise reason — an open cup is contractible (B₂ = 0) and a ring is a flange attached to the
stem (B₁ = 0). The diagnostic difference is **geometric** (the shape and curvature of the base), not
topological; Betti numbers are the wrong invariant even with a flawless 3D scan. Torsion is identically 0
(an orientable solid in ℝ³ has torsion-free homology). The only non-trivial topology appears for a *closed*
button-stage veil (B₂ = 1) — the irreducible-ambiguity stage, not the mature-volva diagnostic. This
corrected our own earlier loose claim that the volva yields a B₂ cavity: it does not, and computing it
showed so. Every angle — pixels, latents, 3D shape, and now shape *topology* — lands on the same
information limit; this one lands there provably.

*Empirical check with a real LRM.* To close the loop on real data, we ran a real single-view reconstruction
model (TripoSR) on actual photographs and computed persistent homology on the output density field
(`lrm_betti.py`, `lrm_persistence.py`). The reconstructions are coherent single blobs (largest connected
component ≈ 100%) but their density fields carry *hundreds* of internal cavities and loops (persistent H₂
≈ 480–980, H₁ ≈ 1300–2400) — NeRF texture, not anatomy. A naïve reading finds "separation" (cavity-count
AUROC 0.80), but this is the project's signature trap in a new costume: the H₂-count and H₁-count correlate
0.99 (a single undifferentiated blob-complexity axis), the dominant cavity barely separates (0.72) and the
single most-persistent cavity belongs to an *Agaricus*, and the effect is confounded with appearance (the
large white *Amanitas* reconstruct as denser blobs than a brown field mushroom). It is an incidental
correlate, not the volva — and the synthetic lab proves it *cannot* be the volva (an open cup contributes
B₂ = 0). So the empirical LRM result is over-determined: TripoSR is too noisy to isolate the volva, and the
volva is topologically invisible regardless. (Caveats: one low-fidelity LRM, no segmentation, n = 8 per
class — but the synthetic proof carries no caveats.)

*Morse theory — the finer lens, and the most precise statement of the wall.* Betti numbers missed the volva
because an open cup is a *cancelling* critical pair (homologically trivial). Morse theory counts the critical
points including the cancelling pairs, so it should see what homology cannot. We confirmed this
(`morse_volva.py`): with identical Betti numbers ([1,0,0]), the superlevel (top-down) height-function Morse
filtration registers the cup rim — the *Amanita* gains **+1 H₀ and +12 H₁ critical pairs the *Agaricus* lacks**.
Morse theory genuinely sees the volva where Betti is provably blind. But the same computation pins down *why*
the feature is so hard, in a single number: the volva's critical pairs have persistence ≈ 3–4 against an object
z-extent of ~47 — a **~9 %-of-scale, low-persistence feature, low-persistence by mathematical necessity** because
it cancels in homology. That is the unifying characterization of the diagnostic feature: topology misses it (it
cancels), global pooling washes it out (low signal), the real LRM buries it (hundreds of higher-count
reconstruction-texture pairs), capture loss removes it (occluded in ~⅔ of photos), and it is single-modality
(no independent confirmation). Morse theory does not break the wall; it states it exactly — the volva is a
*real, geometric, low-persistence, often-uncaptured, single-modality* signal, which is the precise profile every
lever in this paper failed against.

*Can generative modeling synthesize the missing channel?* Two natural proposals: (a) **predict the microscopy
from the macro** and decide on it, or (b) invert — **generate synthetic macros from the diagnostic micro** and
diagnose with those. Both are bounded by the same fact, the data-processing inequality. (a) A predicted
microscopy is f(macro) and carries no more edibility information than the photo; on the confidently-wrong
poison the predicted signal collapses to the edible value (`predict_micro.py`: corroborating with predicted
micro = 2.2 % vs macro-alone 2.3 %), because microscopy is diagnostic precisely by being *orthogonal* to gross
morphology — unpredictable exactly where it is needed. (b) A *faithful* micro→macro generator must reproduce
the real overlap of lookalikes, so a classifier trained on its synthetic macros matches the baseline
(`synth_macro.py`: 8.0 % vs 5.3 %, no gain); a generator that *does* separate them is hallucinating a
difference the photos do not carry, and it scores a perfect 0.0 % on synthetic test while eating 9.0 % of real
Destroying Angels — the aggregate-separability trap, now *pre-validated* and therefore maximally dangerous. The
one genuine payoff is that the micro-grounded densities let you **measure** the irreducibly-ambiguous region and
abstain on it principledly (decide the confident half at 2.8 % residual). Generation is a tool for calibrating
abstention and covering rare species — not for separating lookalikes, because at inference the query specimen's
micro was never measured, so the decision stays bounded by the one channel in hand.

*Can AI upscaling reveal the missing detail?* No, and for the same reason, now shown empirically and
adversarially. AI super-resolution is f(low-res photo) → high-res photo, so it adds no edibility information
(data-processing again); it re-renders plausible high-frequency texture from a natural-image prior. We
deliberately downsampled the lethal pair, upscaled back, and re-encoded with the same frozen I-JEPA
(`upscale_test.py`): Swin2SR x4 lands *at or below* bicubic and below the original (15-seed: original
0.949/6.9 %, bicubic 0.945/8.9 %, Swin2SR 0.933/9.8 %), and an adversarial panel that re-ran harsher
downsampling found it degrades *monotonically* (SR−bicubic AUROC −0.012 at 56 px → −0.045 at 16 px; SR
false-eat up to 24.4 %), while a *stronger* generative super-resolver (a latent-diffusion x4 upscaler) was
worse still (0.884/19.2 %) — because a more aggressive prior injects shared hallucinated texture across both
classes, *raising* the catastrophic false-eat. (For this pair the gross morphology mostly survives
downsampling, so the claim is about *recovery and ordering*, SR ≤ bicubic ≤ original, not the magnitude of
loss.) The diagnostic micro-features — basidiospores at 5–15 µm — are below a phone's optical resolution at
field distance, so they are not in the captured photons at all; no function of the image can reveal them. The
sharper image is the danger, not the cure: it merely *looks* more diagnostic, inviting false confidence.

**8.3 What data collectors must do differently.** *Observation/reporting:* always excavate and photograph the
**full stem base** (the single highest-value fix); standardized multi-view (cap, gill underside + attachment,
profile, base, cross-section); **substrate/host tree** (the causal ecological covariate that is *null* in
current data — our metadata signal was species-range confound, not edibility); spore print; bruising reaction;
smell; scale + color reference; and honest uncertainty. *Technology:* phone LiDAR + multi-view → 3D;
macro + focus-stack → fine detail; clip-on multispectral/UV → chemistry/fluorescence; miniature e-nose → VOCs.
*ML:* train multi-modal encoders **on these channels** with the asymmetric cost + conformal abstention
developed above — the architecture was never the bottleneck; the **modality** was.

**The directed positive.** Better capture and a part-cascade *raise the ceiling and move the abstention
boundary* substantially; they do not zero the irreducible core. The honest program is therefore constructive,
not defeatist: **stop squeezing a richer answer out of an impoverished capture — collect the channels that
carry the missing physics, deliberately localize the diagnostics that are present, and abstain honestly on the
rest.** That is exactly how a real mycologist works, and it is fully consistent with the one lever that ever
moved the needle.

---

### Appendix — artifacts (all runnable, `~/Documents/EatThisMushroom/`)

`real_ijepa_test.py` (the empirical battery: separability / decision / open-set / crop / augment) ·
`evaluate_signals.py`, `gate.py` (divergence floor + symmetrization-null standing gate) ·
`value_of_channel.py` (independent vs redundant channel) · `net_benefit.py` (decision-curve analysis) ·
`conformal_predict.py` (any-toxic-in-set + open-set break) · `cv_smote.py` (LOSO + SMOTE) ·
`gen_ood_test.py` (generative OOD) · `metadata_fusion.py`, `geo_confound.py` (species-range confound) ·
`fgvr_head.py` (metric-learning decoupling) · `ot_domain_adapt.py` (OT vs CORAL) ·
`tda_persistence.py` (PomSeg cubical PH) · `epistemic_voi.py` (EVPI active inference) ·
`vlm_ask_directly.py` (frontier VLM, the canonical question) ·
`hwm_forager.py`, `forage_expedition.py` (the two cautionary rigged builds, kept as negatives) ·
`FINDINGS.md` (the short companion).
