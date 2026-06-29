# Should I eat this mushroom? — honest findings

A research vehicle for studying **when a learned system should refuse to act**, built on the canonical
JEPA question. Not foraging advice. Under any uncertainty the answer is **DON'T EAT / consult a
mycologist** — and the central result below is *why* that is the only safe answer.

The headline: across ~15 methods, **the limit on this problem is information, not method sophistication.**
Almost nothing in the modeling toolbox helps; the one thing that ever moves the needle is an *independent
sensor*, which is a fact about data, not a clever algorithm.

---

## What genuinely holds (the real contributions)

| finding | what it says | artifact |
|---|---|---|
| **The coincidence / divergence floor** | On feature-coincident clouds (a poison whose latent equals an edible's), *every* f-divergence and Wasserstein → 0 simultaneously. It's an information floor (MI=0 on a fixed feature set), not a measure not-yet-tried. | `evaluate_signals.py`, `gate.py` |
| **Value of an independent channel** | Only a channel whose errors are *independent* of the image drives false-eat → 0 — and it must be ~0.95+ reliable (a real toxin/spore/DNA assay). A redundant image-like channel, however accurate, barely helps. | `value_of_channel.py` |
| **Decision-curve analysis** | At the field-standard lethal cost ratio (c_PSC=100, = FungiCLEF), **no operating point beats "eat nothing."** The statistician-legible statement of "no point is both safe and useful." | `net_benefit.py` |
| **Conformal prediction** | "Any-toxic-in-set → don't eat" gives a real distribution-free guarantee on *known* classes — but the guarantee is **void for open-set novel toxins**, exactly where danger is highest. | `conformal_predict.py` |
| **The real empirical battery** | On real I-JEPA latents, a 0.94 "separability" coexists with an unfixable ~5% confidently-wrong false-eat on real Destroying Angels, and ~chance novel-toxic detection. | `real_ijepa_test.py` |
| **VLM knowledge ≠ solving the floor** | A frontier VLM (Qwen3-VL-32B) asked the real question refuses all 80 deadly specimens and names the Amanita 96% of the time — but that's *world knowledge* of famous species, on curated photos, not a solution to open-set or coincidence. | `vlm_ask_directly.py` |

These are genuine because they survive scrutiny — several were nearly the *opposite* of what looked true at first
(the 0.94 separability, the metadata "0% false-eat," the "+5% epistemic" all dissolved under fair tests).

---

## What does NOT help — and we checked, hard

| method | verdict | how we know |
|---|---|---|
| more data (300→stable, FROZEN encoder) | no | residual is representational; a *head* on frozen latents can't fix it |
| **trained domain encoder + 4.5× data** (ResNet-50, 16 species; multi-seed) | **helps in-category, NOT novelty** | vs frozen I-JEPA on the *same* data: lethal false-eat 3.1/8.1% → **0.4±0.3 / 3.8±0.8%** (non-overlapping CIs — training genuinely helps, corrects "more data won't help"). BUT the open-set "win" was a **sister-species mirage**: sister *Amanita* 3.3% vs **true-novelty unrelated genera 7.9±0.7% ≈ frozen 7.1%** — training does *nothing* for genuine novelty. Floor persists (residual 0.4–3.8%). Ceiling up, floor stays — FungiCLEF to the decimal |
| FGVR metric-learning head | no | drove aggregate AUROC to 0.99 while false-eat got *worse* |
| cross-validation / SMOTE | no | CV is an estimator; SMOTE fixes imbalance, not coincidence (LOSO reveals open-set 22–82% eaten) |
| Wasserstein / KL / α-β divergences | no | all vanish on the coincidence floor (proved) |
| generative / diffusion OOD | no | counterproductive — OOD AUROC *below* JEPA |
| metadata (GIS/season) fusion | partial→no | mostly species-range confound; collapses under co-occurrence; causal habitat field is null |
| Optimal Transport (domain adapt) | **no** | OT-Sinkhorn *hurts* (AUROC 0.910) vs no-adaptation (0.936) and CORAL (0.931) |
| **HWM hierarchical planning** | **no** | **rigged/misattributed twice** — the apparent win was an objective swap (total-value vs rate), not hierarchy; a flat planner matches it; the route search contributes ≈0 to −0.5 |
| **Active Inference epistemic term** | **no** | inert even with a proper EVPI formulation — natural Bayesian belief-updating already explores enough |
| TDA persistent homology | **real but dominated** | cubical PH (PomSeg method) genuinely beats trivial image stats (0.811 vs 0.631) — but ≪ I-JEPA (0.973), adds nothing on top (0.954), and the latent manifolds are *not* topologically distinct (bottleneck ≈ within-class null) |
| cascade / part-localized (RGB crops) | no (modality, not localization) | 20-seed: GLOBAL 0.56±0.73% vs CASCADE-loc +0.25% (within noise, AUROC +0.001); veto gate harmful @80%. A 5-lens panel built a stronger 10-region part-detector — also no gain beyond seed noise. Regions are ~70–80% predictable from global = **redundant** RGB (value_of_channel). Verified FAIR (not a strawman) |
| dense JEPA feature-field (the JEPA+3DGS proposal's runnable core) | no | 10-seed: mean-pool 0.5/1.6%, vs max 0.7/2.2%, meanmaxstd 0.5/2.0%, **zero-shot part-clustering 0.4/1.5%** — all inside each other's CIs. Dense/part-structured I-JEPA patch features do **not** beat global pooling; the local-volva-via-patches claim doesn't survive on RGB. The full Feature-3DGS adds only multi-view distillation (infeasible — internet photos are different single-view specimens) and occluded-part *inference* (hallucinates the volva from the cap = the deadly residual re-added). Localization/aggregation limit ruled out; it's an **information** limit |
| 3D topology / Betti numbers (B₀/B₁/B₂, torsion) | **no — provably** | gudhi-validated 3D lab: mature *Amanita* (open volva cup) and *Agaricus* have **identical Betti numbers [1,0,0]** — an open cup is contractible (B₂=0), a ring is an attached flange (B₁=0). The volva is **geometric, not topological**; Betti is the wrong invariant even with a perfect 3D scan. Torsion ≡ 0 (orientable solid). Occluded base → same [1,0,0]. (Corrected our own earlier "volva → B₂" hand-wave by computing it.) |
| 3D topology — EMPIRICAL (real LRM: TripoSR + persistent homology on real photos) | no (confound, not anatomy) | `lrm_betti.py`/`lrm_persistence.py`: TripoSR reconstructs real photos, persistent homology on the density field. Apparent separation (cavity-count AUROC 0.80) is a **trap = the 0.94-separability trap in 3D form**: it's *hundreds* of NeRF-texture cavities (not one volva), H₂-count ≈ H₁-count (r=0.99 → undifferentiated blob-complexity), dominant cavity barely separates (0.72) and *Agaricus* holds the most-persistent one; confounded with appearance (white/large Amanitas → denser blobs). Provably not the volva (synthetic lab: open cup B₂=0). One low-fi LRM, n=8 — caveated, but the synthetic proof is caveat-free |
| **Morse theory** (height-function critical points) | **finer than Betti — sees the volva, but doesn't break the wall** | `morse_volva.py`: identical Betti [1,0,0], but the **superlevel (top-down) Morse filtration registers the cup rim** — Amanita gets **+1 H₀ and +12 H₁ critical pairs Agaricus lacks** (the "cancelling pair" homology is blind to). Vindicates the instinct AND characterizes the difficulty in one number: the volva is a **~12%-of-scale, LOW-persistence** feature *by necessity* (it cancels) → buried among the LRM's hundreds of reconstruction-texture pairs, still needs the volva captured, still single-modality RGB. Morse theory gives the most precise statement of the wall, not a way through it |
| predict **micro from macro** (spores/basidia/tissue) | **no — provably (data-processing)** | `predict_micro.py`: predicted-micro = f(macro) carries no more edibility info than the macro (DPI); 20-seed, corroborating with PREDICTED micro = 2.2% ≈ macro-alone 2.3%, and on the confidently-wrong poison the predicted signal is +0.97 (≈edible, fails) vs MEASURED +2.32 (flags it). Microscopy is diagnostic *because* it's orthogonal to macro → unpredictable exactly where needed |
| invert: **micro→macro generation** → synthetic macros → diagnose | **no for separation; YES for abstention calibration** | `synth_macro.py`: faithful generator reproduces the real overlap → synth-trained = baseline (8.0 vs 5.3%, no gain); a generator that *does* separate is **hallucinating** → 0.0% on synthetic-test but 9.0% on real (the 0.94→death trap, pre-validated = most dangerous). Genuine payoff: micro-grounded densities **measure the ambiguous region** → principled abstain (confident half 2.8%). At inference the query's micro is never measured → decision stays bounded by the macro |
| **AI super-resolution / upscaling** of the macro photo | **no — provably (data-processing)** | `upscale_test.py`: downsample the lethal pair then upscale + re-encode (same frozen I-JEPA). Swin2SR x4 lands **≤ bicubic, below original** (15-seed: orig 0.949/6.9%, bicubic 0.945/8.9%, SR 0.933/9.8%) and degrades **monotonically** with harsher downsampling (SR−bicubic AUROC −0.012@56px → −0.045@16px; SR false-eat → **24.4%**). A **stronger** generative SR (SD x4 latent-diffusion) is worse still (0.884/**19.2%**). SR = f(low-res) adds no edibility info (DPI) — it re-renders plausible detail from a prior; the *sharper image invites false confidence*. **Caveat:** this pair's macro-morphology mostly survives downsampling, so the claim is about **recovery/ordering** (SR ≤ bicubic ≤ original), not magnitude of loss. **Adversarial panel: fair, not strawman; holds & strengthens under harsher loss + stronger SR** |

---

## The methodological lesson (the part I have to be honest about)

Building the HWM application, **I twice produced a result that looked like a win and was not** —
`hwm_forager.py` (a rigged baseline + an inert hierarchy) and `forage_expedition.py` (a real objective
improvement mislabeled as "hierarchy helps"). Both were caught by **adversarial-verification workflows**
(panels of skeptics that read the code, ran their own counter-tests, and decomposed the effect). The lesson
is blunt and it generalizes: **motivated reasoning toward a positive result is real, and the verification
discipline — not the author's initial "it works" — is what keeps the science honest.** Every load-bearing
claim here was put to a panel or a fair ablation; the ones that survived are above, the ones that didn't are
in the "does not help" table.

---

## Bottom line

"Should I eat this mushroom?" from a photo is bounded by **information**, not by the cleverness of the model.
A confident classifier, a sharper representation, a better divergence, a hierarchical planner, optimal
transport, an exploration bonus — none of them manufacture the discriminating feature (the volva) that the
photo or the frozen encoder does not contain. The only thing that works is acquiring a genuinely independent
observation (spore print, chemistry, microscopy) — which is why a real mycologist uses several, and why the
honest output of a system like this is *abstention*. **Research vehicle, never a forager. When in doubt, DON'T EAT.**
