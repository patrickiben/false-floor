# Reading *A False Floor*: an educational companion

## Start here

This guide explains a study about how machine-learning researchers can mistake a model’s limitation for a limitation of the information available to every model. It is for readers who can work with percentages but are new to representation learning and statistical bounds. It is a teaching interpretation of a preprint, not a new experiment or a food-safety guide.

The central distinction is simple: a photograph can contain a useful clue that one encoder fails to retain. Failure after that encoder therefore does not, by itself, show that the photograph lacks the clue.

**Safety boundary:** none of these results establishes that a wild organism is safe to eat or handle. A model’s research score cannot clear an individual specimen. Under uncertainty in real foraging, abstain and consult a mycologist, as the paper’s ethics statement says.

### Which paper this guide follows

The basis is Patrick Iben’s *A false floor: Refuting our own information-limit claim in a lethal identification task*, **Zenodo v2, record 23578222**, deposited in October 2026. The [source record](https://doi.org/10.5281/zenodo.23578222) contains the main preprint and S1 Appendix. Its 90-page preprint file combines 65 main-text pages and the 25-page appendix. Page locators below use the printed main-text pages or the appendix’s own 1–25 numbering.

This guide follows that version, including its corrections, measured training-catalog overlap, field audit and post hoc analyses. Earlier TMLR and workshop teaching copies are not its evidence basis. The historical `companion/MANUSCRIPT.md` in this repository is the original investigation being criticized, not this educational guide.

## 1. The vocabulary and the question

An **encoder** turns a photograph into a list of numbers called an **embedding** or **representation**. A **frozen encoder** keeps its pretrained weights fixed during the new study. A **probe** is a smaller model trained to predict the new target from those numbers. Frozen does not mean that the whole pipeline is untrained: the probe still learns from labeled training examples.

The **baseline** is the comparison pipeline. The original investigation used I-JEPA; the main counterexample uses DINOv2. **PCA**, or principal component analysis, compresses a representation into directions of high variation. High variation need not be the same as useful variation for a particular decision.

A **false-eat rate** is the fraction of poisonous specimens that the rule clears. **Edible yield** is the fraction of edible specimens it clears. Both are conditional rates within a class. Neither is the probability that a particular cleared specimen is poisonous; that further depends on which organisms a user encounters.

A **floor** is a proposed lower limit: at a specified yield, no image-based rule could reduce false-eat below it. Such a claim is much stronger than saying that every method tried so far failed. **Bayes error** describes an ideal best-possible rule for a specified distribution and observation; it is not an error rate automatically identified by one fitted model.

**AUROC** summarizes ranking over many thresholds. It does not directly answer the paper’s decision question, which asks how many poisonous images are cleared at a specified edible yield.

**Where we are in the source:** Introduction; Materials and methods, “The claim under test” and “Statistical analysis” (main pp. 2–12); S1 §A (pp. 1–2).

**What the passage does:** it separates a claim about one representation from a universal claim about photographs.

**In ordinary language:** failing to read a clue through one pair of glasses does not prove that the clue is absent.

**Evidence status:** this is the study’s conceptual distinction and task definition. The analogy is teaching material, not an experimental result.

**Do not accidentally conclude:** a better encoder proves that every difficult photograph is identifiable, or that a practical identification service is safe.

**Try it:** if two classes coincide after compression, what additional measurement would distinguish “compression discarded the clue” from “the original photograph never contained it”? Answer after the exercises.

## 2. How the comparison works

The main mushroom comparison holds the cached photographs, random split procedure, probe family and target yield fixed while changing the encoder pipeline. Each of ten seeds divides the available image pool into 70% training and 30% testing. The probe uses 48 principal components fitted on the training split and an ensemble of 12 small neural networks. The encoders use their own standard preprocessing.

At the main operating point, the eat threshold is chosen using the held-out **edible** scores so that empirical edible yield is 80%. Held-out poison labels do not set the threshold. This estimates a point on the sampled discrimination frontier; it is not a threshold that a deployment team can simply install with a guaranteed future 80% yield.

The original floor came from a different absolute-threshold rule on one split, read through I-JEPA and a 64-component projection. Its most permissive reconstructed yield was 70/89, or 78.7%; other criteria gave lower yield. A comparison must discuss this constraint rather than silently treating every threshold rule as identical.

A separate calibration exercise splits held-out edible images into calibration and evaluation halves. The decisive poison count stays zero in every seed, but realized evaluation yields range from 61.7% to 90.0%, averaging 79.2%. This is why “target 80%” must not become “guaranteed 80% on future images.”

**Where we are:** Materials and methods, “The claim under test”; Results, “The counterexample,” including its operating-point and held-out-calibration discussion (main pp. 5–10, 16–29); S1 §J.

**What the passage does:** it specifies what is held constant and what remains different.

**In ordinary language:** compare two readers on the same examination, while checking that one has not been allowed to reject more questions.

**Evidence status:** these are reported study procedures and calibration results. The guide has not retrained the encoders or reproduced the image experiments.

**Do not accidentally conclude:** this comparison isolates the training objective alone. Encoder architecture, pretraining corpus and readout also differ. Nor are the ten seeds ten independent samples of organisms.

**Try it:** explain why a rule that abstains on every image has zero false-eat but does not solve the stated 80%-yield task.

## 3. The result that carries the argument

The paper starts with the original **7/81** false-eat count. Its exact one-sided 95% lower confidence bound is **4.13%**. The authors compare a candidate counterexample’s upper bound with that lower bound, rather than treating the approximate “5% floor” as known without uncertainty.

| Question | Reported count or result | Interpretation in this version |
|---|---|---|
| Honey mushroom versus deadly Galerina, DINOv2 | 0/150 in every seed; per-cell one-sided 95% upper bound 2.0% | The load-bearing counterexample to the broad task-level floor |
| Original pair (field mushroom versus Destroying Angel), ten-split DINOv2 | Worst seed 2/150; upper bound 4.14% on the recorded reference system | Does not fall below the floor’s 4.13% lower bound; those ten-split counts decide nothing about a floor on that original pair |
| Original pair, additional out-of-fold analysis | 0/500, with each Destroying Angel held out once; reported bound 0.60% | Added after the ten-split result was known; not part of the main refutation |
| Wider comparison | DINOv2 has a lower seed-mean rate than I-JEPA in all 21 plotted pairs | Descriptive breadth, not 21 independent interval-based refutations |

The decisive pair is **not the original pair**. It answers the broader claim that the sampled photographic task has a floor that every image-based rule must respect. If the claim were narrowed to one particular original pair or life stage, evidence from another pair would not settle it.

For zero events in n independent binomial observations, the one-sided 95% upper bound is:

`upper = 1 − 0.05^(1/n)`

For n = 150, that is about 1.98%, reported as 2.0%. Zero observed errors therefore does not mean zero possible risk. A confidence bound is a repeated-sampling statement under its sampling assumptions; it is not a 95% personal safety probability for the next organism.

The ten seeds repeatedly reshuffle one image pool. Do **not** replace n = 150 with n = 1,500. A specimen can occur in several test splits. Nonzero table cells are seed means, and their parenthetical numbers are descriptive diagnostics produced from rounded means, not confidence intervals for those means. Even a displayed 0.0 requires a check of integer counts before the paper calls it an observed zero.

### A qualification that must stay beside the headline

The paper did not fix a multiple-comparison family in advance. A Bonferroni adjustment over six encoders raises the decisive zero’s upper bound to about 3.1%; over 15 label-free hard-pair cells it is about 3.7%; over all 30 admissible hard-pair cells it is about 4.2%, above 4.13%. The numerical separation therefore depends on the declared family. The guide does not present the result as an unconditional, family-independent proof.

The post hoc 0/500 count also does not supply 500 independent prospective tests of one frozen deployed rule. Each specimen is held out once within an out-of-fold procedure; the fitted rules share training data. Preserve the paper’s exploratory status and its decision not to make this count load-bearing.

**Where we are:** “Statistical analysis”; Results, “The counterexample,” especially Table 2; Fig. 1; Limitations (main pp. 10–12, 16–29, 41–43, 53–54); S1 §§B, J.

**What the passage does:** it identifies which integer count can challenge which claim, with uncertainty and selection qualifications.

**In ordinary language:** one successful counterexample can break a statement about every method, but only at the scope and uncertainty that the counterexample actually supports.

**Evidence status:** reported counts and bounds; the formula above is an explanatory calculation. Small platform-dependent changes in some nonzero counts are documented in S1 §J. The all-seed decisive zero is reported to reproduce across those platform checks.

**Do not accidentally conclude:** a rounded zero, a seed average or a long list of winning rows is automatically the same evidence as the audited count.

**Try it:** why do 4.14% and 4.13% matter even though both round to 4.1% at one decimal?

## 4. Strong performance and training provenance answer different questions

A pretrained model may carry useful general visual knowledge, have encountered an evaluation photograph, or have learned labels closely related to the new target. These possibilities require different evidence.

The study runs eight encoders on the five hard mushroom pairs. Six lack declared taxonomic-label supervision. Of those six, only I-JEPA, DINOv2 and Web-SSL are **label-free** in the paper’s stricter sense: no text or label channel. CLIP, SigLIP 2 and MetaCLIP 2 use paired text, which could contain species names. They provide corroboration, not the load-bearing label-free counterexample.

BioCLIP 2 and 2.5 perform extremely well, but the paper excludes them from the information-in-the-photograph argument because they use hierarchical taxonomic supervision and declare sources including GBIF, from which the evaluation images were acquired. A species name determines the task’s edible/poisonous label by a lookup. This is an evidentiary exclusion, not a statement that BioCLIP performs poorly or that its performance was caused by memorization.

The new version actually checks the published training catalogs. It recovers identifiers for **2,775 of 3,600 scored images**. Among those recovered images, **95/2,775** appear in both pinned catalog revisions: exact two-sided 95% interval **2.78–4.17%**. All 95 matches lie on the original control pair; the decisive pair has **0/884 recovered** matches. The term “no-image-overlap” is the study’s threshold category, whose upper bound is at most 5%; it does not literally mean that no image overlapped.

Recovery is incomplete, especially for earthballs, where only 39/400 images were recovered. Identifier joins also do not reach re-hosted EOL copies. Catalog absence is not proof of absence from every training channel or the actual training run. DINOv2’s training images were not joined against a public catalog. Its lack of labels is not proof of image disjointness.

**Where we are:** Materials and methods, “Training-catalog overlap”; “Eight encoders, four pretraining objectives”; “The domain models, measured,” Tables 4–5 (main pp. 12–13, 30–40); S1 §§E–F, J.

**What the passage does:** it separates declared provenance, measured image overlap, label exposure and performance.

**In ordinary language:** finding an answer key in a training source is a different question from finding the exact examination photograph there.

**Evidence status:** the counts measure recovered-image matches against published catalogs. They do not experimentally identify what caused a model’s accuracy.

**Do not accidentally conclude:** zero recovered decisive-pair matches clears every leakage concern, or 95 matches prove that a model memorized those examples.

**Try it:** a photograph was taken after a fixed training snapshot. What can that date rule out? If it was taken before the snapshot, what does the date alone establish?

## 5. What the broader examples and literature audit add

The plotted study includes 21 pairs across five biological groups, organized as six decisions because plants and berries are separate decisions within the plant group. DINOv2’s seed-mean rate is lower in every plotted pair. These are useful stress tests of the original attribution, but species recur, some image pools differ, and small samples matter. Three frog pairs have only 22–30 held-out specimens. On one, DINOv2’s own estimate is 10.0%. Two Müllerian frog comparisons involve two defended species and test species separability, not “safe versus dangerous.”

Fig. 1 places dots at seed means and draws bars toward parenthetical bounds or diagnostics. Those bars do not all share the same inferential status. Do not count the 15 bars that fall below the old approximate floor as 15 independent refutations.

A separate literature audit asks a narrower question: in a citation-defined sample, did a paper score a biodiversity model on images from a declared training source without a qualifying provenance check? It does **not** measure actual contamination in every positive paper.

Of 24 coded eligible papers, **8 were positive, 11 negative and 5 unknown**. The positive share is 33.3%, with exact 95% interval 15.6–55.3%. The prespecified reading is **indeterminate**, not “common.” Counting every unknown as positive gives a sensitivity bound of 13/24. The “rare” threshold was unattainable at this sample size even if no paper were positive; the paper discloses that asymmetry.

The screener, two coders and operator were instances of the same language model. Their agreement is model consistency, not independent human reliability. A later different-model check was neither blind nor human. Much of the citation frame was unreadable to the automated routes, and the internal dated plans have no external timestamp. These are limitations of the audit, not missing footnotes to discard when teaching its result.

**Where we are:** Replication and Fig. 1; “Whether evaluations in one citation frame check provenance,” Table 7; Limitations (main pp. 40–45, 53–54); S1 §§B, I.

**What the passage does:** it shows where the concern recurs and what the audit can actually estimate.

**In ordinary language:** a warning sign about evaluation practice is not a measured diagnosis of every model’s training contamination.

**Evidence status:** observed benchmark comparisons plus a small model-coded documentary audit; neither is a population-wide survey of all biological AI.

**Do not accidentally conclude:** the audit proves that one third of all published biodiversity results are invalid.

**Try it:** rewrite “33% of papers were contaminated” using the actual estimand and uncertainty.

## 6. Carrying the method into another study

The theoretical statements remain useful. If an observation truly contains no class information in the relevant regime, a transformation cannot manufacture the missing information. But a lossy encoder can discard information that the photograph contains. Showing overlap in its feature space does not establish overlap in the original observation space.

S1 §D offers six checks: try the strongest plausible baseline; change representation while holding the question fixed; test the theorem’s antecedent; collect data for the claimed difficult regime; attach count-based bounds; and state the conclusion under the real loss and operating constraint. Passing these checks still does not certify a universal impossibility.

**Hypothetical transfer example:** a team says a blurry archive cannot support date classification because one text-recognition pipeline fails. Before declaring the date unknowable, compare an appropriate stronger reader on the same scans and date groups, audit training overlap, define the rejection rate and evaluate genuinely difficult scans. If the new pipeline succeeds, it refutes the broad claim for those sampled scans; it does not recover illegible text from every damaged page. No new empirical result is asserted by this example.

The paper’s four mechanisms explain why a believable story can still fail: a weak baseline, a curated proxy, a true theorem with an unmeasured condition, and verification aimed at fabricated wins rather than overclaimed losses. Wider PCA projections improve I-JEPA, so the readout contributed to its residual. Aggregation also matters on the spider ablation. The evidence does not isolate one universal failure mechanism for all encoders.

**Where we are:** Discussion, “Four mechanisms” and “Scope of the result” (main pp. 46–54); S1 §§A, C–D, F–G.

**What the passage does:** it turns the self-correction into a reusable research procedure.

**Evidence status:** the six-check procedure is the paper’s methodological recommendation; the archive example is a hypothetical transfer exercise.

**Do not accidentally conclude:** reproduction of an old pipeline proves its broad interpretation, or success on curated images removes deployment uncertainty. The study is not a prospective field-safety evaluation.

## Exercises

1. Define false-eat and edible yield using their denominators. Why are they not the same as “chance a cleared specimen is poisonous”?
2. Why is the all-abstain rule excluded from the main comparison?
3. Use `1 − 0.05^(1/n)` to compare zero errors in 60 and 150 observations. Which zero provides a tighter bound?
4. Ten seeds each test 150 images from overlapping splits. Is 1,500 the number of independent tested specimens? Explain.
5. The original pair’s upper bound is 4.14% and the proposed floor’s lower bound is 4.13%. Does the comparison separate them?
6. Why is the 0/500 out-of-fold result not the paper’s main refutation?
7. What changes when the correction family expands from six encoders to 30 cells?
8. Explain why the category “no-image-overlap” is compatible with 95 catalog matches.
9. Why can a post-snapshot date exclude an image while a pre-snapshot date cannot prove its inclusion?
10. Rewrite “the audit proved contamination was common.”
11. Name one finding that narrows the readout explanation and one remaining uncertainty about the encoder pipeline.
12. Transfer: a clinical-image study claims “the image cannot reveal the disease” after three models fail. Specify a fair counterexample experiment and one clinical conclusion that would still be unsupported.

## Answer key

1. False-eat divides poisonous images cleared by poisonous images tested; yield divides edible images cleared by edible images tested. The risk among cleared images also depends on class prevalence and sampling.
2. It clears no edibles, so it does not meet the 80%-yield constraint.
3. The bounds are about 4.87% and 1.98%. The latter is tighter; neither is zero risk.
4. No. The image pool is reshuffled and specimens recur. Preserve per-seed n and dependence.
5. No. The upper bound is slightly above the lower bound, even though both round to 4.1%.
6. It was added after the main result was known, uses a different out-of-fold procedure and is explicitly not load-bearing. It concerns the original pair rather than the decisive Galerina pair.
7. The adjusted zero bound rises from about 3.1% to 4.2%, which no longer falls below 4.13%. The family was not fixed in advance.
8. It is a threshold label based on an interval upper bound at most 5%, not a literal count of zero. Here the pooled measured interval is 2.78–4.17%.
9. A photograph created later cannot be in the earlier snapshot. Earlier existence only makes inclusion possible; a catalog or content join is needed to establish it.
10. In this citation-defined, model-coded sample, 8/24 papers had at least one same-source evaluation without a qualifying check; the 95% interval was 15.6–55.3%, so the prespecified reading was indeterminate. Actual image contamination was not the estimand.
11. Example: changing pooling does not rescue I-JEPA on the mushroom ablations, while wider PCA reduces its residual. Objective, corpus and architecture are not independently identified as causes; DINOv2 image overlap is unmeasured.
12. Hold the image pool, patient split, target sensitivity/yield and supervision rules fixed; test an appropriate stronger representation with an audited count and uncertainty, including provenance controls. Success could challenge the sampled-task impossibility claim. It would not establish patient safety, prospective benefit or performance under every acquisition condition.

Answer to the §1 “Try it” prompt: test a stronger or substantially different representation on the same original images, with the same task, held-out split procedure and yield constraint. Success would show that the first representation discarded usable signal; repeated failures alone would still not prove that the original images lack it.

These are suggested answers for self-study. No learner study, human grading or transfer assessment has been conducted for this guide.

## Source and reproducibility notes

All empirical statements above summarize the preprint, not a new reanalysis. The guide’s small probability calculations are independently recomputed for arithmetic checking. Original data, images, weights and scientific manuscripts are unchanged.

- Main source: [Zenodo v2 record 23578222](https://doi.org/10.5281/zenodo.23578222), file `Iben_2026_A_False_Floor_preprint_v2.pdf`, SHA-256 `ca83ef3cfcedeea504ad20281787b07d7dea04f797c450e9f356d9729bf2a761`.
- Supplement: same record, `Iben_2026_A_False_Floor_S1_Appendix.pdf`, SHA-256 `52d9fb7630123d435747e105f7e520c0281ae691ddb4a4ae21fe89f12c74410d`.
- Catalog and field-audit records cited by this version: [Zenodo record 23569075, version 4](https://doi.org/10.5281/zenodo.23569075). These are records, not the manuscript.
- The paper’s data-availability and S1 §J distinguish replay of retained hashes and joins from re-hashing nonredistributed photographs, and document platform changes in some nonzero counts. This guide does not claim a full image-level reproduction.

Prepared 23 October 2026 with AI assistance as a source-locked educational explanation. Its review covers textual fidelity and teaching structure. It does not certify the preprint’s scientific validity, full G14 completion, accessibility for every reader or human learner validation. This private repository’s visibility is unchanged.
