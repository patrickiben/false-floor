# Complete reading companion to *A False Floor*

## How to read this complete companion

This companion follows the current *A false floor: Refuting our own information-limit claim in a lethal identification task* through the full main text and S1 Appendix. Its exact basis is **Zenodo record 23578222, v2**, not the older TMLR manuscript stored in this private repository. The official combined PDF has 65 main pages followed by the 25-page S1. Main and S1 page numbers below refer to their separate printed numbering. The source PDF SHA-256 is `ca83ef3cfcedeea504ad20281787b07d7dea04f797c450e9f356d9729bf2a761`; the separate S1 PDF SHA-256 is `52d9fb7630123d435747e105f7e520c0281ae691ddb4a4ae21fe89f12c74410d`.

The [concise study guide](STUDY_GUIDE.md) is the shorter route. This complete route develops the argument in smaller units rather than summarizing the paper into a few broad lessons. Each unit identifies its source passage, explains its role and meaning, labels the evidence, develops reasoning, preserves the relevant boundary and supplies a question with an answer. Revisited findings serve different purposes in results, methods, figure-reading and limitation units. The final coverage map accounts for every substantive section, table, figure, formal proposition and the administrative record.

All empirical counts and model results are **reported by the source**, rather than reproduced here. Hypothetical examples and arithmetic illustrations are explicitly labeled. This is an educational explanation, not an independent empirical replication.

### Boundaries that apply to every unit

- This is not food-safety validation or a tool for identifying an edible specimen. The source says to abstain under uncertainty and consult a mycologist.
- A low random-sample error rate does not bound a selected hard specimen, life stage or unlike field photograph.
- The decisive Galerina pair is different from the original pair on which the old residual was measured.
- Ten overlapping seed splits do not multiply the number of independent specimens by ten.
- Seed means and their descriptive parentheses are not automatically exact confidence intervals. Integer counts and their sampling scheme matter.
- Multiplicity-family choice changes the decisive bound; no family was fixed in advance.
- Image overlap, taxonomic-label exposure and model performance are distinct questions. Neither catalog matches nor their absence alone identify the cause of a score.
- The field audit measures reported evaluation practice in a small citation-defined, model-coded sample. It is not a measured contamination rate across all biological AI.

## Framing, task and measurement

### F01. Reading a self-refutation without turning it into a safety success story

**Source locator:** title, Abstract and Introduction, main pp. 1–5; Conclusion, pp. 54–55.

**Passage role:** announces that the author is correcting a previous interpretation and identifies the methodological lesson.

**Plain-language explanation:** The original investigation concluded that photographic identification faced an information limit: some poisonous specimens seemed confidently clearable no matter which tested method was used. The new paper argues that this residual depended on the chosen representation and operating rule. A stronger frozen representation gives a bounded counterexample on a deadly look-alike pair. That changes the interpretation of the earlier failure.

It does not make the new pipeline a safe foraging service. The title’s “false floor” names an apparent lower limit created partly by how the observation was represented and analyzed. It is not a claim that all biological ambiguity has disappeared. The paper also audits domain-model provenance and a sample of published evaluation practices. Those are related checks on interpretation, but they are not the same experiment as the encoder counterexample.

**Evidence type:** a summary of reported experiments and the author’s self-correction. An abstract’s repeated headline is not an independent replication of the count later shown in Table 2.

**Reasoning or worked illustration:** Imagine a hypothetical scanner whose default compression removes tiny lettering. Several readers fail on its output, and the team declares the original document unreadable. A better representation of the same scan can refute that broad declaration without showing that every torn page is readable. The analogy explains the logical move in the paper. It does not claim the biological task behaves exactly like printed text, nor does it establish which visual features a stronger encoder uses.

**Boundary/nonclaim:** the source’s useful result is a narrowed scientific attribution. A zero observed harmful count remains uncertain, and the real-world cost of one error remains unchanged. The companion must not convert a self-correction into permission to eat or handle an organism.

**Self-check:** What does the new result change most directly: the safety of an unknown mushroom, or the interpretation of the old model’s residual?

**Answer:** It changes the interpretation of the old residual. It does not establish the safety of an unknown specimen.

### F02. Why failure of several methods does not prove an information limit

**Source locator:** Introduction, main pp. 2–5; Discussion “Four mechanisms that produced the false floor,” pp. 46–50; S1 §D, pp. 6–7.

**Passage role:** explains the asymmetry between finding a counterexample and establishing an impossibility claim.

**Plain-language explanation:** “Every method we tried failed” describes a finite search. “No image-based method can succeed” quantifies over all suitable rules on the observation. The second statement is far stronger. A collection of methods can share an encoder, a compression, training data or a weak decision rule, making their failures dependent on one common bottleneck.

Trying more downstream methods does not necessarily challenge that bottleneck. The paper therefore asks for the strongest plausible baseline and a substantially different representation. Its argument is deliberately one-sided: one properly scoped successful rule can overturn a universal lower-bound claim, whereas agreement among several failures only establishes a common residual for those methods. More failures can be informative, but the chain from those failures to an information limit requires additional assumptions or direct evidence about the observation itself.

**Evidence type:** logical distinction and methodological recommendation, motivated by the reported counterexample. It is not a theorem that every negative machine-learning result is false.

**Reasoning or worked illustration:** Suppose a hypothetical team tests ten classifiers, all using the same two blurred measurements. Their shared failure could reflect the measurements, or it could reflect what those two features discard from a richer image. Replacing classifier number ten with eleven leaves that uncertainty largely untouched. Testing a representation that retains a different kind of image information attacks the common assumption more directly. Holding the task and evaluation constraint fixed prevents a supposedly successful replacement from solving an easier question.

**Boundary/nonclaim:** success on one sampled distribution does not establish success everywhere. Conversely, the critique does not mean theory or repeated failures are useless; it means the scope of a conclusion must match the evidence and the assumptions connecting it to the modality.

**Self-check:** Why might fifteen failed methods provide less evidence about a photograph-level limit than one comparison with a genuinely different encoder?

**Answer:** If the methods share the same lossy representation, they repeatedly test that bottleneck. A different encoder can directly challenge whether the photograph contained discriminating signal that the shared representation discarded.

### F03. The old investigation is a failure specimen, not the current paper

**Source locator:** Materials and methods “The claim under test,” main pp. 5–8; S1 §C, pp. 5–6; Data availability, main pp. 57–58.

**Passage role:** preserves the original claim and method history so the self-refutation can be audited.

**Plain-language explanation:** The archived `MANUSCRIPT.md` is the investigation whose interpretation is being challenged. It contains the original sweep, decision apparatus and claims. The current source describes about fifteen methods, grouped in S1 into thirteen families. The families include more data, metric learning, resampling, divergence/transport objectives, augmentation, metadata, conformal methods, planning, active inference and several geometric or generative approaches.

Merely listing them does not demonstrate that each independently measured the information in the original image. S1 also notes fine-tuning results and alternative yield-matched readings that already suggested lower residuals, but whose surviving evidence is the archived document rather than a run log. The archive reports fine-tuned ResNet means of 3.8% at 80% yield and 0.4% at 60%, with standard deviations 0.8 and 0.3 percentage points; its same-acquisition frozen comparison is 8.1% and 3.1%. Other archived I-JEPA yield-matched means are 1.6% at 80% and 0.5% at 60%. These historical values lack surviving run logs and do not become new audited counts. Preserving that distinction keeps the historical record from being retroactively made cleaner or stronger than it is.

**Evidence type:** quoted historical investigation and a source-level account of its surviving artifacts. These are provenance facts about what was reported, not a fresh rerun of every candidate method.

**Reasoning or worked illustration:** A hypothetical laboratory notebook can document that a result was observed even when the original instrument file is missing. It is still weaker reproduction evidence than a preserved script, inputs and output log. Calling the notebook an “archive” does not restore the missing file. Similarly, retaining the old manuscript allows readers to inspect the error without endorsing its conclusion. A corrected paper can responsibly preserve material it now disputes.

**Boundary/nonclaim:** do not use the old manuscript as current educational guidance or treat its unsupported original interpretation as a new finding. The repository’s historical snapshot is not silently updated by adding this companion.

**Self-check:** Why should a repository keep an incorrect original claim rather than replacing its text with the new conclusion?

**Answer:** The preserved specimen makes the correction auditable. Replacement would obscure what was actually claimed, which methods were tried and which records survive.

### F04. False-eat and yield have different denominators

**Source locator:** Materials and methods “The claim under test,” main pp. 5–9; S1 §A definitions, p. 1.

**Passage role:** defines the decision problem before comparing error rates.

**Plain-language explanation:** The rule has two outputs: clear the specimen or abstain. False-eat is the fraction of poisonous specimens it clears. Edible yield is the fraction of edible specimens it clears. Both are class-conditional rates. They do not directly give the probability that a cleared specimen is poisonous.

That further depends on the mixture of edible and poisonous organisms encountered and on whether the benchmark represents it. A rule that always abstains has zero false-eat but also zero edible yield. The study therefore compares false-eat at a target edible yield rather than rewarding unlimited abstention. On non-ingestion tasks the analogous name is false-safe, with task-specific definitions. The two defended-species frog cases need special handling because their rate measures species separability rather than dangerous-versus-harmless clearance.

**Evidence type:** task definition. A precise denominator is part of the estimand, not an optional reporting style.

**Reasoning or worked illustration:** In a hypothetical test with 100 edible and 100 poisonous images, a rule clearing 80 edibles and two poisons has 80% edible yield and 2% false-eat. Among its 82 clearances, however, two are poisonous: about 2.44%. If the encountered class mixture changes, the risk among clearances changes even when the two conditional rates stay fixed. These numbers are teaching arithmetic, not the paper’s observations. They show why a quoted 2% conditional rate cannot be pasted onto an individual forager’s situation as a personal risk estimate.

**Boundary/nonclaim:** the word “eat” is a research decision label, not advice that any output authorizes consumption. Neither an attractive yield nor a high AUROC establishes an acceptable real-world loss.

**Self-check:** What is wrong with interpreting “false-eat 2%” as “every specimen labeled edible has a 2% chance of being poisonous”?

**Answer:** The denominators differ. False-eat conditions on poisonous specimens; risk among cleared specimens conditions on the model’s clearances and depends on prevalence and sampling.

### F05. The counterexample’s scope depends on the claim’s quantifier

**Source locator:** Results “The one-sided falsification logic,” main pp. 22–29; Table 2, p. 18; Scope and Conclusion, pp. 51–55; S1 §A.

**Passage role:** explains why the decisive pair differs from the pair that produced the original residual.

**Plain-language explanation:** The original count concerns field mushroom versus Destroying Angel. The load-bearing new zero concerns honey mushroom versus deadly Galerina. That change matters. A lower-bound claim about every image-based rule for the broad sampled lethal-identification task can be challenged by a suitable bounded rule within that claimed scope.

A narrower claim about only the original species pair, a particular hard life stage or an observation selected for ambiguity would not be decided by another pair. The paper interprets its counterexample against the broader task-level attribution and explicitly reports that the original pair’s ten-split evidence does not beat the old floor’s own lower bound. The companion must preserve both statements. Otherwise a reader may think the new zero directly repeats the exact original biological comparison when it does not.

**Evidence type:** logical scope plus reported experimental distinction. The argument’s force is conditional on what the refuted claim actually quantified over.

**Reasoning or worked illustration:** A hypothetical statement says “no document scanner can read any serial number below this contrast level.” One correctly bounded successful scanner on an included document challenges that universal statement. It would not decide a different statement saying “this particular damaged serial number is unrecoverable.” The point is not that a convenient example wins every debate. The example must genuinely belong to the claim’s scope and satisfy its operating constraint. That is why species pair, target yield and observation distribution remain part of the teaching record.

**Boundary/nonclaim:** the decisive result is not proof of a lower error rate on every pair. It also does not erase uncertainty about individual specimens or every reasonable multiplicity adjustment.

**Self-check:** What additional evidence would be needed to refute a limit expressly restricted to the original field-mushroom/Destroying-Angel distribution?

**Answer:** A properly scoped rule and uncertainty comparison on that original distribution, under an appropriate yield constraint. A result on the Galerina pair alone would not settle that narrower claim.

### F06. The original 7/81 count is not the same as the approximate five-percent slogan

**Source locator:** Materials and methods “The claim under test”; Table 2 and notes, main p. 18; Results operating-point discussion; S1 §§C and J.

**Passage role:** ties the old floor to a specific count, rule, split and realized yield.

**Plain-language explanation:** The original investigation reported an approximate residual near 5% across repeated runs, but that summary lacks the integer counts needed for an exact bound. The auditable reference used here is seven poisonous clearances among 81 tested on one split. Its point fraction is about 8.64%, while its one-sided 95% lower confidence bound is 4.13%. The lower bound, not the raw fraction or the approximate slogan, becomes the comparison for a counterexample’s upper bound.

The original rule also had different realized edible yields depending on its criterion. Its most permissive reconstructed yield was 70/89=78.7%, while stricter versions gave lower yield. The original pipeline used frozen I-JEPA and a 64-component projection, not the new study’s uniform 48-component rule. These details prevent an inflated story in which all original and new scores are assumed to share a perfectly identical estimator.

**Evidence type:** archived count and reported reconstruction, with independently checkable arithmetic. The approximate old summary is historical context, not a source of additional independent binomial trials.

**Reasoning or worked illustration:** An estimated lower limit is uncertain. If a proposed counterexample has an upper bound below 5% but above 4.13%, it has not achieved the paper’s stricter interval-separation criterion. The lower bound lets the comparison acknowledge uncertainty in the claim being challenged. It does not make the old estimate exactly true or prove that its sampling assumptions hold in the field.

**Boundary/nonclaim:** do not call 7/81 “a 5% observed error rate.” Do not compute an exact interval from the rounded multi-run mean. Platform checks later change some reconstructed nonzero counts, which the source preserves rather than hiding.

**Self-check:** Why compare a counterexample’s upper bound with 4.13% instead of merely with the approximate 5% label?

**Answer:** The old floor is itself estimated. Comparing with its lower bound applies the paper’s more conservative separation criterion and avoids treating the approximate estimate as known without uncertainty.

### F07. What GBIF photographs represent, and what they leave out

**Source locator:** Materials and methods data and species acquisition, main pp. 5–10; Replication qualifiers, pp. 42–43; Scope, pp. 51–54; S1 §§B and J.

**Passage role:** describes the sampled observation distribution and its relation to biological and field claims.

**Plain-language explanation:** The experiments use real organism photographs acquired through GBIF occurrence and species endpoints. They are commonly research-grade records selected because an organism was visible enough to document. That creates a useful benchmark but not a random sample of all photographs a forager might take. Species, life stages, backgrounds and photographic quality can be uneven. Most acquisition takes one image per occurrence.

The small Allobates collection is an exception, using multiple media from some occurrences; those images can share a biological event. Labels come from the record’s taxonomy rather than a prospective ingestion experiment. The study compares encoders within a pair using the same cached pool wherever possible. Two coral-snake comparisons have smaller harmless pools for I-JEPA, and the source discloses that exception. Such details constrain how strongly one can attribute a difference to representation alone.

**Evidence type:** reported observational sampling and acquisition. The data are real, but “real” does not mean representative of every deployment regime or free of clustering and labeling error.

**Reasoning or worked illustration:** Suppose a hypothetical archive favors clear full-organism pictures, while a user submits one cropped cap or leaf. An encoder can perform well on the archive because contextual shape and visible parts help. That result still shows useful signal in the sampled photographs. It does not establish the same error bound for the isolated part. The mismatch concerns the population of observations, not whether the test images are genuine photographs.

**Boundary/nonclaim:** these are species-distribution comparisons, not guarantees for a chosen specimen or life stage. The photographs are not redistributed by the educational addition, and their rights do not become unrestricted merely because GBIF provides an API.

**Self-check:** Can a curated benchmark simultaneously make a strong encoder look safer than deployment and a weak encoder’s failure look more fundamental than it is?

**Answer:** Yes. Clear curated images can favor a capable model, while persistent failure of a weak representation on those images can appear misleadingly stable and intrinsic.

### F08. Frozen encoders still lead to a learned decision pipeline

**Source locator:** Materials and methods model/probe description, main pp. 8–10; Results encoder comparison, pp. 30–33; S1 §§G and J.

**Passage role:** identifies the fitting pipeline and the elements held common across encoder comparisons.

**Plain-language explanation:** Each image becomes a cached embedding from a frozen pretrained encoder. Within each seed, the available images are split 70/30. The training split supplies the PCA directions and a small ensemble probe. The common main protocol retains 48 principal components and trains twelve neural-network members, with per-member bootstrapping. The encoders use their own canonical image preprocessing.

Frozen therefore describes encoder weights, not the entire analysis: the probe and training-split projection are learned. Holding the probe family and evaluation rule common makes comparisons informative, but a common dimensional bottleneck need not retain equal task information from every encoder. The paper later shows that widening PCA improves I-JEPA. The complete “encoder pipeline” includes architecture, pretraining corpus and objective, feature readout, preprocessing and compression. A result from swapping that pipeline does not isolate one of these causes by itself.

**Evidence type:** reported protocol and paired design. The companion does not train the ensemble or regenerate its feature caches.

**Reasoning or worked illustration:** In a hypothetical two-feature embedding, the highest-variance direction might describe background and a low-variance direction might distinguish species. PCA keeps variance, not guaranteed decision relevance. Two encoders can therefore lose different information when reduced to the same number of components. Equal numerical dimensionality is a useful control but not proof of equal effective task capacity. This explains why the later width sweep matters rather than treating it as an optional technical curiosity.

**Boundary/nonclaim:** a learned probe may succeed because the original representation exposes useful signal, but the experiment does not identify a human-readable diagnostic trait or a unique causal pretraining mechanism. It also does not fine-tune every encoder to its maximum achievable performance.

**Self-check:** What two fitted elements remain even though the image encoder is frozen?

**Answer:** The training-split PCA projection and the downstream ensemble probe. The threshold-selection rule is another explicitly defined part of the decision procedure.

### F09. A threshold chosen at eighty-percent yield is a frontier measurement

**Source locator:** Materials and methods decision rule, main pp. 8–10; Results “The counterexample” operating-point and held-out-calibration discussion, pp. 22–29.

**Passage role:** defines the operating constraint so a lower harmful-clearance rate cannot be won merely by rejecting more edible images.

**Plain-language explanation:** The main rule places an eat threshold at the held-out edible-score percentile giving empirical 80% edible yield. Held-out poison labels do not choose that threshold. The comparison therefore asks what harmful clearance accompanies a common empirical yield on the sampled test distribution. This is a useful measurement of a discrimination frontier.

It is not the same as choosing a threshold on training data and deploying it with a guarantee of 80% future yield. Using held-out edible information is part of the stated evaluation design, not a secret leak the reader must discover. The interpretation changes when that information would not be available at deployment. A separate calibration exercise examines this issue rather than pretending empirical and prospective yield are identical. The original absolute-threshold rule also has measured yields below or near the new target, so constraint matching requires explanation.

**Evidence type:** reported operating-point definition and a limitation of its deployment interpretation. No prospective field threshold is validated here.

**Reasoning or worked illustration:** In a hypothetical set of 100 labeled edible images, selecting a threshold to clear 80 fixes test-set yield by construction. On another 100 edible images the same threshold might clear 70 or 90. Both facts are compatible: empirical tuning met its target on the first set, while future sampling changes realized performance. If a model clears fewer poisonous images only because it clears almost no edibles, it has not solved the same constrained task. Conversely, a fair constrained comparison must not silently change the label information available when setting its threshold.

**Boundary/nonclaim:** empirical 80% is not a guarantee about the next user, region or image source. It is also not evidence that an 80%-yield operating point is acceptable under lethal loss.

**Self-check:** Why does excluding held-out poison labels from threshold choice matter, yet still not make the threshold directly deployable?

**Answer:** It prevents optimizing the harmful outcome on its test labels. But the rule still uses labeled held-out edible scores to set the test operating point, information a future deployment may not have.

### F10. A seed mean is not an observed binomial count

**Source locator:** Statistical analysis, main pp. 10–12, printed lines 250–313; S1 §§B and H.

**Passage role:** separates descriptive table summaries from the count-based evidence that carries the falsification.

**Plain-language explanation:** Each seed supplies a false-eat fraction. The displayed rate averages those fractions over ten overlapping reshuffles of one image pool. A specimen can recur in multiple test splits, so the seeds are dependent. The archived scripts form a parenthetical by rounding mean-rate times n to an integer and passing it to a Clopper–Pearson routine. For nonzero means, that integer is an arithmetic construction rather than an observed count from a particular seed.

The parenthetical is therefore a descriptive scale check, not a confidence interval for the mean. A nonzero mean can even round to an integer zero and print the same bound as a true zero count. Exact zero-event language requires the archived per-seed counts to show zero clearances. Precision of printing is a consistency check, not the evidence by itself. The special pooled ordinary-pair and large open-set summaries can round small nonzero rates to zero.

**Evidence type:** source-defined statistical reporting rule. The companion does not endorse treating repeated, dependent fractions as independent trials.

**Reasoning or worked illustration:** A hypothetical ten-seed study with n=150 has one harmful clearance in one seed and none in the other nine. Its mean fraction is 1/1,500, or about 0.0667%, which prints as 0.1% at one decimal. Multiplying that mean by 150 gives 0.1, which rounds to zero for the diagnostic routine. That derived zero is not “zero events in every seed.” This illustration explains the distinction using an invented count pattern, not an additional paper result.

**Boundary/nonclaim:** neither a fractional mean nor a narrow-looking parenthesis automatically supplies a valid interval. Multiplying n by ten would overstate independent specimen information.

**Self-check:** If a table says 0.1% with the same parenthesis as a zero-event bound, may it be described as an exact observed zero?

**Answer:** No. Inspect integer counts. The nonzero seed mean and rounding procedure can produce that parenthesis without an all-seed zero.

### F11. What a zero-event confidence bound says

**Source locator:** Statistical analysis and Table 2, main pp. 10–12 and 18; Results undersampling correction, pp. 18–20; S1 §D, pp. 6–7.

**Passage role:** prevents “zero observed” from becoming “no harmful risk exists.”

**Plain-language explanation:** Under the stated binomial sampling model, observing zero events in n observations yields a one-sided 95% upper Clopper–Pearson bound `1 − 0.05^(1/n)`. For 150 observations it is about 1.98%, printed as 2.0%. For 60 it is about 4.87%. The second zero is less informative because a larger harmful rate remains compatible with the small sample.

Exact refers to the interval construction under its assumptions, not an assumption-free safety guarantee. The main counterexample reports a per-seed integer count over unique held-out poisonous specimens. Ten split repetitions do not turn its per-cell bound into a bound based on 1,500 independent organisms or into a joint guarantee over every selected analysis. Those are separate dependence and multiplicity questions. Even a correctly constructed random-sample bound does not apply automatically to a deliberately chosen difficult specimen.

**Evidence type:** reported statistical construction and independently reproducible arithmetic. The existence of the formula does not verify the biological sampling model.

**Reasoning or worked illustration:** For zero events, the probability of seeing no events at rate p is `(1−p)^n`. Setting that probability to .05 and solving for p gives the formula above. This explains why the upper bound falls as n grows and why zero never becomes literal zero risk at finite n. It also shows what is assumed: an appropriate repeated-trial model at a common event rate. Correlated photographs, selected specimens or distribution shifts require additional care.

**Boundary/nonclaim:** “95% upper bound” is not a 95% probability that the next mushroom is safe. Nor is a lower confidence bound on the old count the same quantity as this upper bound on a counterexample.

**Self-check:** Why can the exact same observed proportion, zero, have different evidential force at n=60 and n=150?

**Answer:** Sample size changes which nonzero event rates could plausibly produce no observed events. The larger sample excludes a wider range of harmful rates under the stated model.

### F12. Multiplicity changes whether the bound separates the old floor

**Source locator:** Results falsification and correction-family discussion, main pp. 22–29; Limitations, p. 53, printed lines 1388–1392.

**Passage role:** discloses selection uncertainty that a single impressive cell does not capture.

**Plain-language explanation:** When many candidates are examined, a favorable cell can arise partly through selection. The source applies Bonferroni sensitivity calculations to the zero of 150. For a family of m comparisons, the adjusted zero bound is `1 − (.05/m)^(1/150)`. With six encoders it is about 3.1%; with fifteen label-free hard-pair cells about 3.7%; with thirty admissible hard-pair cells about 4.2%.

The last value exceeds the old count’s 4.13% lower bound, so the paper’s interval-separation criterion no longer holds under that broader family. No family was fixed in advance. A guide that reports only 2.0% and calls the refutation universally family-robust would overstate the current source. This qualification does not make the count disappear; it changes the inference one can attach to a selected counterexample.

**Evidence type:** reported multiplicity sensitivity and a design limitation, not a prespecified family-wide confirmatory result.

**Reasoning or worked illustration:** Substituting m=30 yields about 4.175%. Substituting m=6 yields about 3.141%. Both use the same zero count; only the allowance for multiple opportunities changes. These are explanatory computations, independently checked against the source’s rounded figures. The source also notes that a zero of 150 stays below 4.13% only for families up to 27. That threshold is a property of this count and comparison, not a general rule for how many machine-learning models researchers may test.

**Boundary/nonclaim:** do not choose the smallest convenient family after seeing the results and call it prespecified. Nor does a broad correction establish that the original modality-level floor is true; failure to refute under one criterion is not proof of the floor.

**Self-check:** What remains supported if the thirty-cell corrected upper bound is above 4.13%?

**Answer:** The observed zero and descriptive model gap remain. The claimed interval separation under that thirty-cell family is not established; this does not validate the old universal floor.

## Provenance and audit design

### F13. Recovering the identities of the photographs actually scored

**Source locator:** Training-catalog overlap methods, main pp. 12–13, printed lines 314–333; S1 §§F and J.

**Passage role:** explains how the overlap analysis moves from a declared shared source to a measurable catalog join.

**Plain-language explanation:** The original scraper stored photographs under sequential filenames without occurrence identifiers. To test whether those scored images appear in a published training catalog, the study must recover their identities. It repeats GBIF queries, computes perceptual hashes for candidate images and scored photographs, assigns matches within a Hamming-distance threshold of six, and confirms aspect ratios. It then joins recovered occurrence keys, URLs and iNaturalist photo identifiers against two pinned TreeOfLife catalog revisions.

This is an identification procedure, not simply a search for the same species name. The two revisions correspond to BioCLIP 2 and 2.5 and cannot be assumed interchangeable because catalog membership changed between them. The join reaches GBIF and iNaturalist-hosted rows. Re-hosted copies on EOL are outside what the identifier method can match.

**Evidence type:** reported recovery and catalog-measurement procedure. Hash thresholds and aspect-ratio checks support correspondence; they do not create complete recovery by definition.

**Reasoning or worked illustration:** In a hypothetical archive, two photographs show the same species but only one is a training image. A species-level join would mark both and answer the wrong question. An image or source-identifier join is more specific. Conversely, the same photograph can be re-hosted under a different identifier, so a missing identifier match does not prove no copy exists. The method’s strength and blind spot follow from the unit it joins.

**Boundary/nonclaim:** an unavailable image is unknown, not confirmed disjoint. The catalog is a published record of training membership, not direct observation of the model’s training execution. The source later reports an initial error that hashed a later reacquisition rather than the scored images.

**Self-check:** Why is preserving the original occurrence identifier at acquisition preferable to recovering it later?

**Answer:** It keeps the evaluation image tied to its source record and reduces reliance on re-query order, image availability and matching assumptions. The source’s recovery effort illustrates the cost of losing that provenance.

### F14. The overlap categories are decision rules, not literal descriptions

**Source locator:** Training-catalog overlap methods, main pp. 12–13, lines 334–360; Table 5, p. 37; S1 §I timeline.

**Passage role:** declares the overlap estimand, threshold labels and consequence of the measurement.

**Plain-language explanation:** The measured quantity is the fraction of scored images from the five hard pairs found in the published catalogs. The source’s labels are threshold-based. “Measured-overlap” requires an exact interval lower bound at least 50%; “no-image-overlap” requires an upper bound at most 5%; other cases are partial. Therefore the phrase no-image-overlap can accompany a nonzero number of matches. It does not literally assert none.

The planned primary reading pools the five pairs. Pair-by-pair applications and worst-case columns are added after the result. When fewer than half of one species’ images are recovered, the source limits the reading to the recovered subset. A conventional worst case treats every unrecovered image as present; a bounded worst case uses retained ordering and occurrence-key information to rule out some possibilities. These are different assumptions, so both are reported.

**Evidence type:** internally dated analysis plan, reported measurement rules and post-result sensitivities. Internal dating lacks an external timestamp, which the source acknowledges.

**Reasoning or worked illustration:** A hypothetical sample with three matches and an upper confidence limit of 4% would receive this study’s no-image-overlap category despite having three matches. The category means “low enough under this rule,” not logical absence. Likewise, a worst-case calculation is not a new observation of the missing images. It asks how the answer changes under a stated unfavorable assignment of unknown cases.

**Boundary/nonclaim:** the threshold is the study’s reporting choice, not a universal definition of harmless pretraining overlap. Passing it does not remove label exposure, re-hosting uncertainty or unknown training-run details.

**Self-check:** Can the pooled “no-image-overlap” label coexist with a control pair showing meaningful matches and a poorly recovered earthball subset?

**Answer:** Yes. Pooling, pair-specific concentration and incomplete recovery answer different questions. The source presents all three rather than letting the pooled label erase them.

### F15. What population the literature audit samples

**Source locator:** Field audit methods, main pp. 13–16, printed lines 362–441; Results, pp. 44–45; S1 §I.

**Passage role:** defines the documentary sampling frame so its percentages are not generalized to all biological AI.

**Plain-language explanation:** The audit starts from 146 works linked through OpenAlex to BioCLIP or BioCLIP 2 within the stated date window. It uses a seeded screening order and written eligibility rules. Fifty works cannot be read through the automated public routes; 96 are read. Screening first identifies 26 eligible works, one is later found ineligible, and one of the remaining 25 cannot be coded after becoming unreachable, leaving 24 coded papers.

These stages are different denominators. The final positive fraction concerns those coded eligible papers, not all 146 works, all citations, all benchmark evaluations or the entire biological-AI literature. Inaccessibility is not random by assumption. Publisher blocks, source availability and inclusion rules can shape which documents remain. The audit’s bounded sampling frame is a strength for reproducibility and a limitation for population-wide prevalence claims.

**Evidence type:** reported documentary sample construction and model-assisted screening. It is not a probability sample of every relevant publication.

**Reasoning or worked illustration:** If a hypothetical review finds eight positives among 24 readable eligible papers, dividing eight by 146 treats unreadable and ineligible works as confirmed negatives. Dividing by 25 while counting an unreachable coding outcome as known also changes the estimand. The correct denominator follows the question and missingness policy. A flow diagram or explicit ledger is more informative than choosing whichever fraction looks most reassuring or alarming.

**Boundary/nonclaim:** an inaccessible paper is not evidence of misconduct or a negative result. The frame does not estimate how often all published information-limit claims fail a stronger-baseline test; the source explicitly says that would require a different survey.

**Self-check:** Why should the 50 unreadable works remain visible when interpreting the final 8/24 result?

**Answer:** They limit coverage and may differ systematically from the coded sample. The observed fraction does not automatically describe their evaluation practices.

### F16. A positive audit code is not measured contamination

**Source locator:** Field audit methods and coding rules, main pp. 13–16; Results Table 7, pp. 44–45; S1 §I per-paper table and notes.

**Passage role:** defines the paper-level event counted by the audit and its uncertainty categories.

**Plain-language explanation:** A positive paper has at least one scored evaluation using a biodiversity foundation model on images from a declared training source without a qualifying provenance check. This is a statement about reported evaluation practice. It does not require a measurement proving that the specific test images were in training. The manual distinguishes positive, negative, unknown and unreachable outcomes and explains what counts as a qualifying check.

A paper can contain many evaluations, while one eligible same-source evaluation makes the paper positive. The audit does not retain enough per-evaluation counts to say what fraction of all evaluations or headline results is affected. Source lineage can pass through an aggregator or platform, so coding sometimes requires more than matching one dataset name. The manual and later adjudication rules document such decisions rather than letting each coder improvise a private standard.

**Evidence type:** rule-based, model-coded documentary classification. It measures a specified reporting event, not actual leakage frequency or the size of any performance inflation.

**Reasoning or worked illustration:** A hypothetical paper tests five models on ten datasets. If one relevant model–dataset cell uses a declared source without the specified check, the paper-level indicator can be positive. That does not imply all fifty cells are contaminated or that its main conclusion fails. Conversely, a developer’s qualifying benchmark hold-out can matter even when the evaluated paper did not run a new overlap analysis. The unit of counting and definition of a check determine the code.

**Boundary/nonclaim:** unknown lineage is not confirmed positive. An omitted score can show provenance awareness without satisfying the estimand about checks among reported scored evaluations. The source separates these interpretations.

**Self-check:** Rewrite “a positive code proves the model memorized the test photographs.”

**Answer:** A positive code identifies at least one same-declared-source scored evaluation without the manual’s qualifying check. It does not measure image membership or demonstrate memorization as the cause of performance.

### F17. Internal precommitment, amendments and post hoc work are different records

**Source locator:** Field audit methods; Limitations, main p. 54; S1 §I timeline, pp. 19–21.

**Passage role:** records when rules were written relative to screening, coding, overlap queries and later analyses.

**Plain-language explanation:** The source supplies internally dated plans, amendments and digests. The audit precommitment fixes an estimand and thresholds before paper drawing; later amendments specify details before later stages. The coding manual and adjudication rules are not all fixed at one initial moment. The overlap plan is written before its query, while the out-of-fold count is planned only after the ten-split result is known.

These distinctions matter more than the blanket label preregistered. A hash can show whether a retained file changed relative to the digest; without an external timestamp, it does not independently prove when the file existed. The source explicitly describes its dates as self-dated and states that some design decisions were delegated to the model operator without separate author sign-offs. The companion must not invent a stronger advance-registration or approval record to make the project look tidier.

**Evidence type:** reported chronology and provenance artifacts, with internally dated rather than externally certified timing. Self-dated provenance is distinct from an externally certified timestamp.

**Reasoning or worked illustration:** A hypothetical researcher writes a plan at noon, examines results at one, and adds a new analysis at two. Hashing both files is useful, but it does not make the two-o’clock analysis preplanned relative to the first results. It can be honestly precommitted before execution while still being post hoc to earlier evidence. The out-of-fold analysis has exactly the kind of distinction a reader should ask about: before its own computation, after the main outcome was already known.

**Boundary/nonclaim:** retrospective timing detail does not automatically invalidate an exploratory analysis. It determines what confirmatory status can be claimed and what selection uncertainty remains. It also does not turn model-generated codes into independently human-verified judgments.

**Self-check:** Can an analysis be committed before it runs and still be post hoc in the study?

**Answer:** Yes. It may be designed after earlier results are known. The relevant chronology concerns both its execution and the evidence that motivated its selection.

## Results, counts and stress tests

### F18. Table 1 pools evaluations; Table H restores the individual pairs

**Source locator:** Results “The counterexample,” main pp. 16–18, Table 1; S1 §H and Table H, pp. 14–16.

**Passage role:** shows the ordinary-pair model gap while explaining why the pooled denominator is not a count of independent organisms.

**Plain-language explanation:** The ordinary mushroom comparison includes field mushroom/Destroying Angel, chanterelle/jack-o’-lantern, morel/false morel, oyster/jack-o’-lantern and puffball/Death Cap button. Table 1 pools their ten-seed rates, with I-JEPA at 1.3%, DINOv2 at 0.1% and CLIP displayed as 0.0%. None of these pooled cells is an observed zero. The stated n=720 counts pair-classifier evaluations, covering 570 unique held-out poisonous specimens because Omphalotus appears as poison in two pairs.

S1 Table H supplies per-pair rates and denominators so the pooled summary does not erase that structure. Four DINOv2 ordinary-pair cells are observed all-seed zeros, but none is used as an additional step of the main refutation. The original field-mushroom pair has a nonzero mean of 0.3%. Pooling describes the gap; Table 2 supplies the specific count-based argument.

**Evidence type:** reported descriptive seed means and per-pair count information. The pooled parenthetical is not a confidence interval over 720 independent specimens.

**Reasoning or worked illustration:** In a hypothetical benchmark, one photograph is tested by two different pair classifiers. That creates two evaluations but only one image. Summing evaluation denominators can be useful for workload or descriptive performance, but it changes the observational unit. Likewise, averaging two proportions can yield a small number that rounds to zero even if errors occurred. The count structure must accompany the pooled rate.

**Boundary/nonclaim:** “CLIP 0.0%” is not “no poisonous image was ever cleared.” Nor do the ordinary pairs validate unseen field conditions or make the hard-pair choice unnecessary. The paper keeps the broad descriptive table separate from its narrow inferential count.

**Self-check:** Why are both 720 and 570 worth reporting?

**Answer:** The first counts pair-classifier scorings and the second unique held-out specimens. Confusing them would overstate independent biological evidence in a pooled binomial interpretation.

### F19. The 906-image novel-toxin test is not the 1,151-image test

**Source locator:** Table 1 and notes, main p. 17; Field and shortcut stress tests, pp. 22–25; S1 §B crop/open-set summary.

**Passage role:** examines out-of-training-species behavior while distinguishing two different open-set evaluations.

**Plain-language explanation:** Table 1 scores 906 images of novel toxic species against each of the five ordinary-pair classifiers. It averages across classifiers and seeds. The reported means are 27.1% for I-JEPA, about 0.5% for DINOv2 and 2.9% for CLIP. The script’s whole-percent display rounds DINOv2’s roughly 0.48% to 0%.

That is a small nonzero mean, not a zero event. The denominator represents repeated scorings: 906 images, five classifiers and ten seeds produce 45,300 scorings, with the same images recurring. A different 1,151-image open set is assessed using the original-pair classifier and has a separately reported worst-seed integer count. Combining their sample sizes or quoting one test’s bound next to the other test’s mean would create a result the source never reports.

**Evidence type:** reported open-set descriptive stress test. It measures behavior on the chosen novel toxins, not all possible unseen dangerous taxa.

**Reasoning or worked illustration:** A hypothetical classifier clears one harmful image in 200 tests: 0.5%. A whole-percent print format may display 0%, yet the event remains real. Formatting changes the visible string, not the count. In a repeated-classifier test, counting each repeated scoring as a new organism would add another distortion. Both issues appear in this table’s reporting notes, making it a useful lesson about tracing printed percentages back to their denominator and count.

**Boundary/nonclaim:** “novel” here means absent from the specified training pairs, not guaranteed novel to every encoder’s pretraining. This test cannot bound every future species or selected ambiguous image.

**Self-check:** A guide sees “DINOv2 0% open set” in an old log. What should it inspect before using zero-event language?

**Answer:** The print precision, exact numerator, image denominator, repeated-scoring structure and which open-set experiment produced the number. Here the 906-image pooled result is nonzero.

### F20. Table 2 identifies the count that carries the counterexample

**Source locator:** Table 2, main p. 18; Results “The counterexample” and Conclusion, pp. 19–29 and 54–55.

**Passage role:** isolates auditable integer counts from descriptive summaries so the central argument has a checkable unit.

**Plain-language explanation:** The decisive row is DINOv2 on honey mushroom versus deadly Galerina: zero of 150 held-out poisonous specimens in each of ten seeds. Its per-cell one-sided 95% upper bound is 2.0%, below the original count’s 4.13% lower bound. The table also lists DINOv2’s zero of 120 on puffball/earthball and corroborating zeros from SigLIP 2 and MetaCLIP 2.

Those text-paired models can carry species-name information through captions, so they do not replace the label-free DINOv2 count as the load-bearing artifact. Other table rows describe the original pair’s nonzero worst seed and open-set behavior. A table of counts is therefore not a list of equally decisive refutations. Each row has an argumentative role, a distinct denominator and a scope that the guide must keep visible.

**Evidence type:** reported audited per-seed counts with exact per-count bounds under the stated model. Multiplicity and overlapping-split qualifications remain active from F10–F12.

**Reasoning or worked illustration:** The zero-bound formula gives approximately 1.977% at n=150. That is below 4.126%, the more precise lower reference from 7/81. This elementary interval comparison explains the source’s central criterion. It does not transform the two separate one-sided procedures into an unqualified universal probability statement, and it does not eliminate the dependence of an adjusted claim on the selected correction family.

**Boundary/nonclaim:** this is not 0/1,500 independent poisonous specimens, a universal safety certificate or a direct same-pair repetition of the original residual. Its interpretation concerns the broader claimed task-level floor on the sampled regime.

**Self-check:** Which three descriptors must accompany “zero of 150” to preserve its meaning?

**Answer:** The encoder and species pair, the per-seed unique-specimen denominator rather than seed multiplication, and the bound/selection/sampling qualifications. Without them the headline can imply a different experiment.

### F21. Why the original pair and the later 0/500 remain separate

**Source locator:** Table 2 note a, main p. 18; Results same-pair and out-of-fold discussion; S1 §J, pp. 24–25.

**Passage role:** refuses to overstate the original-pair result and labels the additional analysis’s timing.

**Plain-language explanation:** On the reference-system ten-split result, DINOv2 clears at most two of 150 Destroying Angels. The exact upper bound is about 4.14%, just above the old floor’s lower bound 4.13%. Both print as 4.1% at one decimal, so rounding can hide the fact that the interval-separation rule is not met. Those ten-split counts neither refute nor establish a photograph floor on this original pair.

The authors later run an out-of-fold count in which each of 500 Destroying Angels is held out once, observing zero and reporting a 0.60% bound. That analysis is planned after the ten-split result is known and carries no step of the main refutation. Its fitted folds share training data; it is not 500 prospective tests of one preselected frozen deployed rule. A careful guide reports the useful exploratory observation without borrowing confirmatory status from the earlier experiment.

**Evidence type:** reported same-pair counts plus a post hoc out-of-fold analysis. The zero-bound arithmetic is independently checkable, while the empirical count is source-reported.

**Reasoning or worked illustration:** Subtracting 4.13 from 4.14 gives +0.01 percentage points. The difference is tiny but its direction decides the stated criterion. Replacing both numbers with their one-decimal display changes the apparent logic. Meanwhile `1−.05^(1/500)` is about .597%, explaining the reported .60% without implying independence of fitted folds.

**Boundary/nonclaim:** the later count cannot be retroactively described as the prespecified basis of the paper. Nor does failing the same-pair criterion validate the original impossibility claim.

**Self-check:** What is the right response to two bounds that look equal after rounding?

**Answer:** Inspect their unrounded values and the decision rule. Here the candidate upper bound remains slightly above the floor’s lower bound, so separation is not achieved.

### F22. Table 3 is an encoder sweep, not forty independent experiments

**Source locator:** Table 3, main p. 21; “Eight encoders, four pretraining objectives,” pp. 28–33; S1 §§G and J.

**Passage role:** shows how the residual changes across representations and hard pairs under a common procedure.

**Plain-language explanation:** Eight encoders are tested on five hard mushroom pairs: the original control, honey mushroom/Galerina, woodtuft/Galerina, honey mushroom/sulphur tuft and puffball/earthball. The first three have 150 poisonous test specimens per seed and the last two 120. I-JEPA’s seed means span 2.1–21.3%; DINOv2’s span 0.0–0.7%. The table groups models by masked-latent, self-distillation, image–text and taxonomic-supervision objectives.

Six are admissible under the source’s no-taxonomic-label criterion, but only three are label-free under its stricter no-text/no-label distinction. The BioCLIP rows are reported and excluded from the information-floor argument. Forty displayed cells do not represent forty independent image collections. They reuse pairs, images and splits. Exact all-seed zeros and nonzero diagnostic parentheses also differ in inferential status.

**Evidence type:** reported paired descriptive comparison with selected audited zero counts. It is not a general leaderboard establishing one architecture’s superiority for all vision tasks.

**Reasoning or worked illustration:** The woodtuft row is an important unfavorable qualification. DINOv2’s mean is 0.7%, but its reference-system worst seed clears three of 150, giving an upper bound about 5.1%. The source reports it rather than claiming every hard pair has a worst-case bound below the proposed floor. A broad table can make a pattern visible while the narrower count table determines what supports a specific inference.

**Boundary/nonclaim:** different objectives also use different corpora and architectures. Grouping rows by objective does not isolate the objective as the sole causal reason for a performance gap. Excluded BioCLIP performance is not treated as poor performance or ignored because it wins.

**Self-check:** Why is it useful to report a strong model’s worst hard-pair count even when the central counterexample is a different zero cell?

**Answer:** It prevents the central success from being generalized to every pair and shows that the same reporting standard applies to inconvenient results.

### F23. Crop and degradation tests move toward field difficulty without becoming field validation

**Source locator:** Results “Field and shortcut stress tests,” main pp. 22–25; S1 §B, p. 4; S1 §F correction 3, p. 12.

**Passage role:** challenges easy-image or background explanations and reports a cache error found within that work.

**Plain-language explanation:** The study applies center crops and selected degradations such as blur, downsampling, occlusion and color changes. These checks ask whether the encoder gap relies entirely on full, clean benchmark views. The ordinary-pair crop uses 55% while the hard-pair crop uses 50%.

A shared cache name initially mixed those fractions for some species; re-encoding at the true 50% crop changes the reported DINOv2 range to 0.0–1.4%. Keeping that correction matters because the same pipeline can silently reuse an artifact produced under a different condition. The source also states that the degradation experiment is run on DINOv2 only and that these stress regimes do not directly test the decisive honey-mushroom/Galerina counterexample under every corruption. A guide should not imply that all headline counts have passed every field stress.

**Evidence type:** reported perturbation robustness tests and a corrected implementation artifact. These are engineered stress conditions applied to existing photographs, not a prospective sample of user photographs.

**Reasoning or worked illustration:** In a hypothetical experiment, two scripts both save `crop.npy`, but one crops 55% and the other 50%. Successful loading does not tell the second script which condition produced the cache. Including the parameter in the filename or validated metadata prevents an old artifact from masquerading as new work. The study’s correction is an example of why provenance belongs at the artifact level, not just in a prose methods section.

**Boundary/nonclaim:** synthetic blur and cropping cannot represent all difficult views, damaged specimens or rare life stages. Robustness of seed means is not an exact worst-specimen guarantee. The correction changes an empirical detail and must not be hidden as mere typography.

**Self-check:** What two questions should be asked before reading a cropped result as support for the main zero count?

**Answer:** Was that exact pair tested under that exact perturbation, and do the cached data/metadata confirm the intended crop or degradation parameters?

### F24. The separate 1,151-image open set and its worst seed

**Source locator:** Table 2 open-set row, main p. 18; Field and shortcut stress tests; S1 §B, p. 4; S1 §§F and J.

**Passage role:** reports a count-based novelty stress result without letting a rounded mean erase harmful clearances.

**Plain-language explanation:** The separate open-set experiment uses 1,151 novel toxic images with the field-mushroom/Destroying-Angel classifier. They are not a new held-out split of the decisive Galerina pair. The source’s reference-system worst seed clears four of 1,151, giving an exact one-sided upper bound about 0.8%. Its mean clearance is 0.17%, which an old whole-percent format printed as 0%.

The honest report includes the integer worst seed, the bound and the nonzero mean. This result is different from the 906-image, five-classifier pooled test in F19. The later platform audit changes some nonzero open-set counts while preserving all-seed zeros in the decisive main cell. The source records these movements rather than claiming every nonzero digit is invariant across environments.

**Evidence type:** reported selected open-set count and descriptive mean, with platform sensitivity. It is not an estimate over the universe of all novel toxic organisms.

**Reasoning or worked illustration:** Four divided by 1,151 is about 0.35%. A worst-seed fraction need not equal the 0.17% mean across seeds; the statistics summarize different things. A confidence upper bound is larger again because it includes sampling uncertainty. Keeping numerator, fraction, average and bound separate prevents three legitimate numbers from appearing inconsistent or being substituted for one another.

**Boundary/nonclaim:** an open-set collection can miss the very species or presentation that fails in deployment. Novel to the pair classifier is not necessarily novel to pretraining. Neither the low mean nor the 0.8% bound authorizes a field-safety conclusion under lethal loss.

**Self-check:** Why can a report simultaneously say “mean 0.17%,” “worst seed four clearances,” and “upper bound 0.8%” without contradiction?

**Answer:** They refer to an average, an observed worst-seed integer outcome and uncertainty around that outcome. Their denominators and inferential roles are different.

### F25. Label shuffling and separate calibration test different possible mistakes

**Source locator:** Results null-control and held-out-calibration discussion, main pp. 22–29; S1 §J script ledger.

**Passage role:** checks that the pipeline can lose the signal and that the empirical-yield rule is not mistaken for a deployment guarantee.

**Plain-language explanation:** A shuffled-label control deliberately breaks the correspondence between training labels and image features. The source reports pooled AUROC about 0.50 and false-eat about 79.2% at the target 80% edible yield. Under no class information, clearing roughly 80% of edibles should also clear roughly 80% of poisons. The expected harmful clearance is not 20%; that would confuse clearance with abstention.

A different control separates threshold calibration from edible evaluation by splitting held-out edibles into two halves. The decisive poison count remains zero in every seed, but realized edible yields range 61.7–90.0%, averaging 79.2%. Ten of twenty calibration directions fall below 78.7%, the original rule’s most permissive reference yield, although each seed has at least one direction at or above it. Those facts limit a claim of prospectively guaranteed equal yield.

**Evidence type:** reported negative control and calibration sensitivity. They answer different questions and neither creates independent new specimens by rearranging the same pool.

**Reasoning or worked illustration:** In a hypothetical label-free random score, a threshold retaining 80% of one class retains about 80% of the other as well. Label shuffling checks whether the pipeline falsely extracts separation when correspondence is destroyed. Calibration splitting instead asks whether a threshold chosen on one edible sample realizes its target elsewhere. A model can pass the first check and still have variable yield under the second. The source’s similar 79.2 numbers summarize different experiments and should not be merged.

**Boundary/nonclaim:** preserving a poison zero does not establish a fixed 80% yield in future use. A random-label null also does not prove that the unshuffled model uses biologically meaningful, human-identifiable traits.

**Self-check:** Why is near-80% false-eat, rather than near-20%, a sensible shuffled-label outcome at 80% edible yield?

**Answer:** With no discrimination, the rule clears the same proportion of both classes. Eighty percent is the cleared fraction; twenty percent is the abstained fraction.

### F26. Parameter count and training objective do not uniquely explain the gap

**Source locator:** “Eight encoders, four pretraining objectives,” main pp. 30–33; Discussion aggregation/projection analysis, pp. 46–48.

**Passage role:** tests simple alternative explanations for the representation gap without claiming to identify its complete cause.

**Plain-language explanation:** I-JEPA ViT-H/14 has about 629 million encoder parameters in the cached configuration, compared with about 302 million for DINOv2 ViT-L/14 and the CLIP image encoder. Thus the weaker performance is not explained by the tested baseline simply having fewer parameters. Larger capacity is not synonymous with better features for this task.

The source also notes that sharing an objective does not produce identical results: Web-SSL and DINOv2 both use self-distillation but differ notably on honey mushroom/sulphur tuft. Architecture, training data, preprocessing and readout remain entangled. The claim supported is that the original representation pipeline was insufficient to anchor the modality-level negative, not that one training objective has been experimentally isolated as the causal culprit. This distinction matters when transferring the lesson to another domain, where the strongest suitable baseline may come from a different model family.

**Evidence type:** reported checkpoint configuration and comparative results. These are observational contrasts between pretrained artifacts, not a factorial experiment holding corpus and architecture fixed.

**Reasoning or worked illustration:** A hypothetical larger dictionary can still omit the technical vocabulary needed for one task, while a smaller specialized description retains it. Size is one property, not a guarantee of task-relevant information. Likewise, two models trained with the same loss on different corpora can learn different regularities. To attribute the full gap to objective alone, a new design would have to hold the relevant other factors constant or otherwise identify their effects.

**Boundary/nonclaim:** the paper does not establish DINOv2’s general dominance over I-JEPA or CLIP, nor that more parameters are harmful. It also acknowledges that the common PCA compression enlarges I-JEPA’s residual, so “encoder-only failure” would be too simple.

**Self-check:** What does the 629-million-versus-302-million comparison rule against, and what does it leave unresolved?

**Answer:** It argues against the simple claim that the baseline lost because it was the smaller tested encoder. It does not identify which objective, corpus, architecture or readout difference caused the performance gap.

### F27. Dated model-card reads are reproducibility inputs

**Source locator:** model-card and selection discussion, main pp. 31–33; S1 §E and Table F, pp. 7–10.

**Passage role:** records what was known about training sources, access and licensing at the study’s stated reads.

**Plain-language explanation:** A model nickname is insufficient provenance. The source records exact checkpoints and dated card fields. Some structured dataset fields are empty even when the card text describes a corpus. The absence of a declared source is weaker assurance than a measured disjoint source. The paper does not report image-level overlap checks for the six admissible general encoders, including DINOv2.

Its dated license audit also distinguishes permissive, noncommercial, undeclared and bespoke terms, plus ungated and manually gated access. DINOv2, the counterexample artifact, is recorded as Apache-2.0 and ungated. I-JEPA is noncommercial but ungated. DINOv3 is not run, and the source explains the access/license constraints recorded at its audit dates. These are source-version facts, not a claim that every external card remains unchanged forever.

**Evidence type:** dated documentary reads and study-selection explanation. Empty metadata is not evidence that no relevant images were used in training.

**Reasoning or worked illustration:** A hypothetical model card says nothing in its structured “datasets” field but names a web crawl in prose. Reading only the API field would incorrectly turn “not stated there” into “not used.” Conversely, a card that names GBIF identifies a place to investigate, not the exact photographs a training run consumed. The study combines declaration checks with a measured catalog join where a catalog exists.

**Boundary/nonclaim:** access and license do not determine scientific admissibility. A permissively licensed model can still be excluded by the study’s label-exposure criterion. An unmeasured general-model corpus also cannot be described as proved disjoint.

**Self-check:** If a card changes after the paper’s recorded read, should the companion silently rewrite the original experiment’s provenance?

**Answer:** No. Preserve the dated record and note the later discrepancy separately. A current metadata change is not a retroactive alteration of the weights, images or code actually used in the reported run.

### F28. Table 4 reports excellent excluded-model scores honestly

**Source locator:** “The domain models, measured,” main pp. 33–35; Table 4, p. 34; S1 §E note a.

**Passage role:** makes the exclusion of strong domain encoders auditable rather than omitting inconveniently good results.

**Plain-language explanation:** BioCLIP 2 and 2.5 are run through the same main protocol as the other hard-pair encoders. Both have reported AUROC 1.00 on all five rows. BioCLIP 2.5 has all-seed observed zero clearances on each hard pair, and BioCLIP 2 has four displayed zero rows and one small nonzero row. These are very strong benchmark results.

The paper nevertheless excludes their cells from its information-in-the-photograph argument under a declared criterion: their hierarchical taxonomic supervision and shared declared source distribution can expose the target label, because species names determine the study’s edible/poisonous assignment. The exclusion is not a claim that the models are weak, inaccessible or inaccurate. It is a restriction on which inference the study is willing to draw from their performance. The new catalog measurement narrows the image-overlap part of the concern without changing the label-supervision rule.

**Evidence type:** reported performance and an explicit evidentiary selection criterion. The criterion is the authors’ study rule, not a universal judgment that supervised biological representations are scientifically useless.

**Reasoning or worked illustration:** In a hypothetical task, the target is “class A versus class B,” and a model was trained to name those classes. Its success can be practically useful while providing a different kind of evidence from a model never given the class-name channel. The authors want to infer something about available photographic information without relying on taxonomic target exposure. Whether another study should impose the same restriction depends on its question.

**Boundary/nonclaim:** the table does not prove memorization caused BioCLIP accuracy. The source does not rescore the control pair on a clean subset, so the effect of measured overlap on its score remains untested.

**Self-check:** Can a model be both the best performer in the table and excluded from one particular inferential argument?

**Answer:** Yes. Performance and evidentiary admissibility are different. The exclusion concerns the source and supervision of the artifact, not its numerical rank.

### F29. Table 5 localizes the measured catalog matches

**Source locator:** “What the join finds,” main pp. 36–38; Table 5, p. 37; S1 §§F and J.

**Passage role:** replaces a broad suspicion of image overlap with a bounded measurement on recovered scored images.

**Plain-language explanation:** Of 3,600 scored hard-pair images, 2,775 are recovered, all from iNaturalist. Ninety-five of those appear in both pinned catalog revisions. The pooled fraction is about 3.42%, with exact two-sided interval 2.78–4.17%, receiving the study’s threshold label no-image-overlap. The matches are concentrated on the original control pair: 95/868 recovered.

The decisive pair has 0/884 recovered matches. Other recovered pairs have no matches, but poor recovery matters, especially earthball, with only 39/400 recovered images. Table 5 supplies ordinary and bounded worst cases rather than treating the missing cases as confirmed absent. Both revisions give the same join counts even though they are distinct catalogs. The measurement therefore distinguishes pairs and recovery coverage instead of treating “GBIF in training” as a completed image-overlap analysis.

**Evidence type:** reported catalog join, recovery counts and sensitivity calculations. It is measured overlap in recoverable catalog rows, not direct observation of the actual training run.

**Reasoning or worked illustration:** Ninety-five divided by 2,775 is approximately 3.42%. Dividing by all 3,600 would implicitly assign unrecovered images a known status and answer a different question. A recovered-subset estimate should state its coverage and examine unfavorable missing-case assignments. This arithmetic example uses the paper’s reported counts but does not independently re-identify any photographs.

**Boundary/nonclaim:** zero of 884 recovered images does not prove literal absence of every decisive-pair copy from every training channel. Re-hosted EOL copies and unresolved photographs remain outside the identifier join. The word “no” in the category is threshold language, as explained in F14.

**Self-check:** Why should the 0/884 decisive-pair result and the 95/868 control-pair result both remain visible beside the pooled category?

**Answer:** Pooling hides concentration. The pair-specific results show where measured matches occur, while their recovery denominators and worst cases keep absence claims bounded.

### F30. Seeing an image and seeing its target label are different channels

**Source locator:** “The distinction that keeps the rule narrow” and “An exclusion on label supervision,” main pp. 38–40; S1 §E.

**Passage role:** explains why the source retains the BioCLIP exclusion after low recovered image overlap on several pairs.

**Plain-language explanation:** An image can enter pretraining without an explicit target label, with incidental caption text, or with systematic taxonomic supervision. Those channels support different concerns. Under the paper’s rule, DINOv2, I-JEPA and Web-SSL are label-free because their stated objectives have no text or label channel. This does not establish that they never saw a GBIF photograph.

CLIP, SigLIP and MetaCLIP use image–text pairs, so captions could contain species names; the source uses them as corroboration rather than the decisive artifact. BioCLIP has systematic taxonomic supervision from declared sources, so a species label can carry the edible/poisonous target through a fixed lookup. Low image overlap does not remove that separate label-exposure ground. The authors therefore keep the exclusion on every pair while withdrawing any implication that shared source alone proved image overlap.

**Evidence type:** source-defined provenance distinction and exclusion consequence. It is not a measured decomposition of how much each channel contributed to learned accuracy.

**Reasoning or worked illustration:** A hypothetical photograph shown without its answer label is not the same training event as the same photograph paired with “Galerina marginata.” A different photograph of that species carrying the name also teaches taxonomic information even after duplicate removal. Conversely, removing the explicit label does not guarantee perfect test independence. The relevant claim must specify which channel it excludes or measures.

**Boundary/nonclaim:** the source does not prove that DINOv2’s training corpus is image-disjoint, nor that BioCLIP’s actual scores arise from memorization. Label-free is an objective/channel classification, not a certificate of clean data or a statement that images themselves cannot contain text.

**Self-check:** Why can image-level deduplication leave the paper’s taxonomic-label concern unresolved?

**Answer:** Other photographs of the same species can still carry the species label that determines the target. Removing one duplicate addresses image membership, not all supervised exposure to the taxonomic decision.

### F31. Dates can rule out membership without proving membership

**Source locator:** Discussion “Declared and measured provenance,” main pp. 50–51; catalog methods, pp. 12–13; Limitations, pp. 53–54.

**Passage role:** explains why declarations, snapshots, image dates and joins contribute different kinds of evidence.

**Plain-language explanation:** The TreeOfLife GBIF snapshot is dated 1 May 2024. Much of the recovered decisive-pair material is newer; the source reports that 883 of its 884 recovered images were photographed in 2025 or 2026. A photograph created after a fixed snapshot cannot be in that snapshot. But a photograph created earlier is only eligible for inclusion; its date does not establish that the catalog actually contains it.

The source’s occurrence-key argument also uses query order and catalog maxima to bound some unrecovered cases. These are structured exclusion arguments, not a universal claim that all low keys are included or every date is perfect. The paper recommends reporting snapshot dates, test-image dates, measured identifier/hash joins when possible, and clean-subset results when overlap exists. It also recommends preserving occurrence identifiers at scrape time.

**Evidence type:** temporal provenance reasoning, measured catalog facts and reporting recommendations. It does not establish the contents of an unobserved actual training execution.

**Reasoning or worked illustration:** A hypothetical book published in 2025 cannot have been physically present in a library’s frozen 2024 catalog. A book published in 2020 may or may not be cataloged. The first date comparison proves exclusion from that fixed catalog; the second requires a membership check. The same asymmetry explains why an image’s early existence is not evidence of leakage by itself.

**Boundary/nonclaim:** creation, posting and indexing dates are distinct and may be imperfect. The source’s result relies on its stated recovered identifiers and retained order. Dates do not reach re-hosted copies automatically or prove which examples changed a model’s behavior.

**Self-check:** What four provenance disclosures does the discussion recommend before treating a strong benchmark score as clean evidence?

**Answer:** Training snapshot date, test-image dates, a measured catalog join where available, and results on a clean subset if overlap is found. Each addresses a different missing link.

## Cross-domain comparisons and reading the display

### F32. Six decisions are organized within five biological groups

**Source locator:** Replication section, main pp. 40–43; Table 6, p. 42; S1 §B Tables A–E.

**Passage role:** shows that representation dependence recurs beyond the original mushroom context without pretending every comparison is equally strong.

**Plain-language explanation:** The replication covers mushrooms, deadly Apiaceae, coral snakes, recluse look-alikes, poison-frog mimicry and deadly berries. These are six decisions within five biological groups because Apiaceae and berries are both plants. The main plot includes five hard mushroom pairs plus sixteen replication pairs, for 21 total. DINOv2’s seed-mean dangerous-cleared or task-specific rate is lower than I-JEPA’s in every plotted pair.

That repeated direction supports the methodological concern, but it is not 21 independent interval-based refutations. Species recur across pairs, sample sizes differ and some comparisons measure a different event. Two Müllerian frog rows involve defended species on both sides and test separability. Two coral-snake rows use smaller harmless image pools for I-JEPA. Such exceptions do not erase the comparisons, but they narrow claims of a perfectly uniform paired experiment.

**Evidence type:** reported cross-domain descriptive comparisons and externally motivated look-alike choices. The external references support the biological or misidentification context, not a new measurement of image indistinguishability.

**Reasoning or worked illustration:** A repeated failure pattern across unrelated tasks is more suggestive than one task alone. Yet using the same deadly species in two pairs creates dependence, and a tiny frog sample contributes less precision than a larger pair. Counting rows gives breadth; it does not automatically give equal evidential weight. The paper’s main refutation continues to rest on one audited bounded count, not a vote among species pairs.

**Boundary/nonclaim:** the study does not show all biological mimicry is easy, every published look-alike is separable or every model is suitable for handling dangerous organisms. Some point estimates remain high and some intervals are wide.

**Self-check:** Why is “all 21 point estimates are lower” compatible with saying the frog evidence is only suggestive?

**Answer:** Direction and precision are different. A lower point estimate based on 22–30 dangerous specimens can still have wide uncertainty and a substantial harmful-clearance rate.

### F33. Figure 1 mixes exact bounds and descriptive diagnostics deliberately

**Source locator:** Fig. 1 and caption, main p. 41; Statistical analysis, pp. 10–12; S1 §B.

**Passage role:** makes the paired pattern visible while specifying what the dots and bars mean.

**Plain-language explanation:** Each dot is a ten-seed average for a species pair and encoder. A bar runs toward that cell’s parenthetical value. When the underlying count is zero in every seed, the parenthesis is an exact per-count upper bound. Otherwise it is a descriptive diagnostic from the rounded mean, not a confidence interval for the average. The same visual grammar therefore carries two statistical statuses, explained in the caption.

The shaded region marks values below the approximate old floor. DINOv2’s point is lower in all 21 pairs, and its bar falls below the old floor in fifteen. That fifteen is a descriptive display count, not fifteen independent interval-based refutations. Small-n frog rows visibly have wide bars. The original control and decisive pair should be distinguished by name rather than inferred from the most favorable-looking dot.

**Evidence type:** visualization of reported estimates and diagnostics. The plotted bars do not create stronger inference than the underlying count and sampling construction.

**Reasoning or worked illustration:** Imagine a hypothetical chart where one bar is a measured uncertainty interval and another is a rough scale benchmark. Giving them identical line shapes can help compare magnitudes, but only a legend preserves their different meanings. A reader who assumes every line is a conventional confidence interval can manufacture interval-separation claims the source explicitly rejects. The correct workflow is dot, denominator, integer-count status, then bar interpretation.

**Boundary/nonclaim:** the graph is not a leaderboard over independent biological experiments and does not identify field-safe operating points. The two Müllerian rows also have a different clearance event, declared in the caption.

**Self-check:** A bar for a nonzero seed mean ends below 5%. Does that alone satisfy the paper’s bounded-counterexample rule?

**Answer:** No. Its parenthesis may be descriptive, the correct comparison is the floor’s uncertain lower bound, and multiplicity/scope still matter. The inferential count is separately identified in Table 2.

### F34. Deadly Apiaceae extend the warning beyond mushrooms

**Source locator:** Replication narrative, main pp. 40–43; S1 §B, Table A, pp. 2–3.

**Passage role:** tests whether an apparent residual recurs in a distinct lethal-ingestion setting.

**Plain-language explanation:** The plant comparisons pair poison hemlock with wild carrot, water hemlock with wild carrot, and poison hemlock with cow parsley. Each uses 90 held-out dangerous specimens per seed. I-JEPA’s seed-mean false-eat rates are 9.0%, 5.8% and 16.2%; DINOv2’s are 0.2%, 0.0% and 0.8%. The direction is consistent with the mushroom concern: a weak representation can leave a substantial residual where a stronger one exposes much more discriminating signal.

The zero row’s exact bound is about 3.3% because n=90, not the 2.0% from n=150. Nonzero table parentheses remain diagnostics. The source’s correction ledger also withdraws earlier interval-separation language based on hand-computed lower bounds from rounded means. A full teaching account must preserve that correction even when the qualitative encoder gap looks large.

**Evidence type:** reported replication point estimates and audited zero status, with explicit limits on nonzero parentheses. These are not new prospective ingestion-safety tests.

**Reasoning or worked illustration:** The zero formula gives `1−.05^(1/90)≈3.27%`. A smaller test set naturally gives a wider upper limit than zero of 150. It would be incorrect to copy the mushroom’s 2.0% bound onto this plant row simply because both display 0.0. This illustrative calculation is independently checkable and teaches why each row needs its own n.

**Boundary/nonclaim:** the study does not claim that a chosen hemlock leaf or partial root can be safely identified. Whole-organism archive photographs, repeated species and dependent splits remain the observation regime. The comparison also does not prove the networks use the diagnostic traits a botanist would use.

**Self-check:** What is the justified significance of the plant results in the broader argument?

**Answer:** They show representation-dependent residuals beyond mushrooms on the sampled plant pairs. They do not multiply the decisive mushroom counterexample into three independent interval proofs or establish field safety.

### F35. Coral-snake comparisons include a paired-data exception

**Source locator:** Table 6, main p. 42; replication qualifiers, p. 43; S1 §B Table B, p. 3.

**Passage role:** connects canonical mimicry to image-distribution separation and discloses unequal harmless pools.

**Plain-language explanation:** The coral-snake comparisons use scarlet kingsnake, milk snake and scarlet snake as look-alikes. DINOv2 has observed all-seed zeros at n=90 on all three rows, with exact per-count bounds about 3.3%. I-JEPA’s seed means range 4.2–22.4%. The source points to external mimicry work to motivate resemblance, not to prove that photographs contain no discriminating information.

Importantly, the I-JEPA caches have fewer harmless images for two rows: 198 milk-snake and 45 scarlet-snake images rather than the 300 in the stronger-encoder caches. That changes the available harmless training and yield-calibration material. The paper therefore qualifies its claim that comparisons are perfectly paired. The dangerous-side n can remain the same while an edible/harmless-side difference still affects the fitted decision problem.

**Evidence type:** reported image-classification comparisons with a documented data-pool asymmetry. The biological mimicry reference and the machine comparison answer different questions.

**Reasoning or worked illustration:** In a hypothetical pair, both models test the same 90 dangerous images, but one learns the harmless class from 30 examples and another from 200. Identical dangerous test n does not make training opportunity or threshold estimation equal. The performance difference remains an observation, but attributing every part of it to the encoder requires caution. This illustrates why a complete methods comparison records both classes, not just the harmful denominator printed beside a bound.

**Boundary/nonclaim:** lower clearance rates do not validate snake-handling decisions. Visual mimicry can remain biologically effective even when a model separates curated species distributions. Human, predator and model decisions need not use the same cues or sampling regimes.

**Self-check:** Why is matching only the poisonous or dangerous test denominator insufficient for a perfectly paired comparison?

**Answer:** The harmless training and calibration pools also shape the learned rule and yield threshold. Different pools can affect performance even with the same harmful test count.

### F36. Recluse misidentification is not the same claim as biological indistinguishability

**Source locator:** Replication narrative and Table 6, main pp. 42–43; S1 §B Table C, p. 3; aggregation ablation, main pp. 46–47.

**Passage role:** uses a documented human-identification problem to show why high human error need not establish image-level absence of information.

**Plain-language explanation:** The spider comparisons pair recluse images with southern-house, wolf and hobo spiders. I-JEPA’s dangerous-cleared means span 10.3–48.9%, while DINOv2 spans 0.0–2.0%. The largest absolute gap is the southern-house comparison: 48.9% versus 2.0%. The cited Vetter account concerns systematic human misidentification through unreliable reliance on a violin marking rather than diagnostic eye arrangement.

Strong machine separation can be consistent with that human-error account. It need not overturn the biological observation that people confuse these spiders. It challenges a different inference: that human or weak-model confusion proves the photograph itself is information-poor. The later pooling ablation shows that DINOv2’s feature readout matters greatly on this pair, making it especially useful for separating encoder and aggregation explanations.

**Evidence type:** reported classifier comparisons plus a source’s interpretation of cited human-identification literature. The guide does not independently test human recognition or establish which spider cue the encoder uses.

**Reasoning or worked illustration:** A hypothetical person uses color alone to distinguish two objects and fails, while a model uses shape and succeeds. The person’s failure is real, but it supports a limitation of that strategy rather than a proof that every visible cue is absent. Conversely, the model’s success does not show that its cue is stable in every cropped photograph. The paper’s spider example illustrates this distinction without making a claim about safe personal handling.

**Boundary/nonclaim:** the 48.9-to-2.0 gap is a seed-mean comparison, not two independent binomial intervals. Its n=90 and its diagnostic parentheses retain the source’s reporting rules. The ablation later uses a different capped image pool and must not be numerically substituted.

**Self-check:** Can a machine result agree with the fact that humans often misidentify a species while refuting a claim of photographic indistinguishability?

**Answer:** Yes. Human confusion and absence of discriminating image information are different propositions; they can have different explanations.

### F37. Small frog samples and defended-species pairs change interpretation

**Source locator:** Table 6 notes, main p. 42; replication qualifiers, p. 43; S1 §B Table D, pp. 3–4.

**Passage role:** preserves the weakest replication evidence and makes the event definition explicit.

**Plain-language explanation:** Four frog pairs use n between 22 and 90. Three have only 22–30 held-out focal specimens, so their uncertainty is wide. On Allobates/Ameerega parvula, DINOv2’s own mean is 10.0%, with a large descriptive parenthesis; it is not below the approximate floor in point estimate. On fantastica/imitator, the mean falls from 43.3% to 0.3%, but n=30 limits precision.

Two Ranitomeya pairs are Müllerian: both species are chemically defended. Their rate counts the mimic cleared as the model species, not a dangerous animal mistaken for a harmless one. The Allobates acquisition also uses several images from some occurrences, so images can share a biological event. The source calls this evidence suggestive rather than confirmatory. Those qualifications are essential rather than a distraction from the larger mushroom result.

**Evidence type:** reported small-sample species-separability and mimicry comparisons. Their biological event and sampling differ from a straightforward edible/poisonous decision.

**Reasoning or worked illustration:** Zero observed events in 30 binomial observations would still have an upper bound around 9.5%. That hypothetical zero is much less precise than zero of 150. A small nonzero seed mean cannot improve that evidential limitation by being visually close to the chart’s axis. If both biological classes are defended, even perfect species classification would not become a safe-handling recommendation.

**Boundary/nonclaim:** do not call every frog row a successful lethal-safety discrimination, omit the 10.0% result or promote a lower mean to a tight confidence statement. Multiple images of one occurrence also cannot be assumed independent organisms without further information.

**Self-check:** What two reasons make the spectacular 43.3%-to-0.3% comparison less decisive than its visual size suggests?

**Answer:** The small n yields wide uncertainty, and the Müllerian event concerns separability between two defended species rather than harmless-versus-dangerous clearance. The large mean difference does not remove either qualification.

### F38. The berry rows show that the weak baseline does not always fail badly

**Source locator:** S1 §B Table E, p. 4; main Table 6 and replication qualifiers, pp. 42–43.

**Passage role:** adds an ingestion setting while reporting a case that is already easy for I-JEPA.

**Plain-language explanation:** The berry comparisons include elderberry/pokeweed, wild grape/moonseed and bilberry/deadly nightshade, each with n=90. I-JEPA’s reported means are 19.9%, 10.8% and 0.2%. DINOv2 has observed all-seed zeros in all three rows, with about 3.3% exact per-count upper bounds. The last pair is important because I-JEPA is already near perfect in point estimate.

The study’s concern is not that a particular baseline must fail every task. It is that a substantial residual in a weak representation can be misread as an inherent limit where that representation happens to be inadequate. When the baseline already separates the data well, there is little apparent floor for another encoder to remove. The table also preserves CLIP results, but these are corroborating text-paired-model comparisons rather than the load-bearing label-free count.

**Evidence type:** reported cross-pair descriptive results and audited zero status. These are sampled photographic classification tests, not studies of consumption outcomes or isolated berry clusters.

**Reasoning or worked illustration:** In a hypothetical benchmark with one easy pair and one hard pair, a weak model can score 99% on the easy pair and poorly on the hard pair. Reporting only the hard pair would exaggerate a universal weakness; reporting only the average could hide the pair-specific problem. The source keeps within-pair comparisons visible. The berry example teaches why interactions between task and representation matter more than a blanket label “weak model.”

**Boundary/nonclaim:** none of these rows authorizes eating berries identified by a model. The task uses often whole-organism photographs, so success need not transfer to an isolated fruit, damaged specimen or unusual life stage.

**Self-check:** Does the easy bilberry/nightshade result undermine the methodological argument?

**Answer:** No. It narrows it. The warning concerns attributing a residual to the observation where a representation is weak, not a claim that the same encoder is poor on every biological comparison.

## What the literature audit finds

### F39. Table 7’s primary reading is indeterminate

**Source locator:** Results “Whether evaluations in one citation frame check provenance,” main pp. 44–45; Table 7, p. 44; field-audit methods.

**Passage role:** applies the declared audit rule to the coded sample and shows sensitivity to uncertain coding decisions.

**Plain-language explanation:** Of 24 coded papers, eight are positive, eleven negative and five unknown. The primary fraction is 33.3%, with exact two-sided interval 15.6–55.3%. The rule calls a result common only if the lower interval bound exceeds 25%; it calls it rare only if the upper bound is below 10%. The primary result is therefore indeterminate.

At n=24, even zero positives would have a two-sided upper bound about 14.2%, so the rare category was unattainable. The source also notes the corresponding limitation at the earlier target n=30. Counting all unknowns as positive gives 13/24, a sensitivity bound that would read common. Changing particular lineage or qualifying-check decisions gives other rows, not replacements chosen for their rhetorical effect. The bottom observation, zero papers running their own qualifying check or reporting a clean subset, is explicitly not a prespecified estimand.

**Evidence type:** reported model-coded documentary audit with exact interval arithmetic and sensitivity analyses. Its category rule is asymmetric at the available sample size.

**Reasoning or worked illustration:** The observed 33.3% lies above 25%, but the lower confidence bound does not. A point estimate crossing a threshold is not the same as an interval satisfying the rule. Similarly, unknown cases do not become observed positives because an unfavorable bound assigns them that way. A bound asks how much the reading could change under that assignment.

**Boundary/nonclaim:** do not report “contamination was common” or “the audit proved one third of all papers were invalid.” The measured event, frame, missingness and model judgment remain part of every percentage.

**Self-check:** Why does 8/24 not satisfy the stated common rule despite a point estimate above 25%?

**Answer:** The rule requires the exact lower bound above 25%; here it is only 15.6%. The uncertainty, not just the point estimate, determines the label.

### F40. Agreement among model coders is not human reliability

**Source locator:** audit methods and Results, main pp. 13–16 and 44–45; S1 §I per-paper table, timeline and later-check discussion, pp. 16–21; Limitations, p. 54.

**Passage role:** describes coding consistency, adjudication and the limits of a later second-model check.

**Plain-language explanation:** The screener, two coders and operator are instances of the same language model. Their paper-level codes agree on 23 of 25 papers, with κ=.88, but this includes an agreement on an unreachable paper excluded from the final 24. Such agreement measures consistency under that model and manual, not independent human reliability. The operator’s own known-answer examples contained errors that were corrected against paper texts.

A later Qwen check covers the thirteen positive and unknown papers, agreeing on ten, with κ=.54. It is neither human nor blind because it had read a draft describing the positive codes. The source discusses three disagreements in detail. A folder with a human-coding name does not turn the returned model codes into human judgments. The companion must preserve that distinction instead of treating labels on files as evidence of who performed the work.

**Evidence type:** reported agreement statistics and model-assisted adjudication. There is no human gold-standard coding study here.

**Reasoning or worked illustration:** Two hypothetical copies of the same calculator can agree because they share the same implementation error. Agreement is still useful for detecting inconsistent inputs, but it does not independently certify correctness. A different calculator with the expected answer visible can also agree without providing a blind check. This analogy explains why model identity, shared instructions and prior exposure matter alongside κ.

**Boundary/nonclaim:** disagreement does not automatically make the final code wrong; source-level adjudication can be informative. But neither corrected disagreements nor high same-model agreement justify inventing a human signoff or claiming the audit is free from systematic model error.

**Self-check:** What should a guide say if a file is named as though it contains human codes but the source says a model produced them?

**Answer:** Identify the actual model origin and the nonblind check. The filename is historical provenance, not evidence of human coding.

## Mechanisms, formal statements and the proposed procedure

### F41. Aggregation can matter greatly without explaining the mushroom gap

**Source locator:** Discussion mechanism (a), main pp. 46–47; Table 8, p. 46; S1 §B and its count-audit notes.

**Passage role:** tests whether an unfair feature readout, rather than the pretrained representation, created the comparison.

**Plain-language explanation:** I-JEPA has no class token, so the main readout averages patch tokens. DINOv2’s main readout uses its CLS token. The ablation compares I-JEPA mean/max pooling with DINOv2 CLS/mean/max. It caps species at 250 images, leaving 75 harmful test specimens per seed, so its values are not the same measurements as the full-image tables.

On the spider pair, DINOv2 changes from 2.1% under CLS to 25.7% under mean pooling and 5.5% under max. Aggregation is therefore genuinely lossy there. On the two mushroom pairs, DINOv2 remains at 0.0–0.4% across readouts, while I-JEPA is not rescued by max pooling. The source’s conclusion is conditional: mushroom readout choice does not explain away the main gap, but aggregation is not universally harmless. That distinction prevents the method from being reduced to a slogan that only encoder identity ever matters.

**Evidence type:** reported controlled readout ablation on a smaller image pool. Per-seed audits identify its displayed zeros, but the paper does not make new zero-event refutations from this table.

**Reasoning or worked illustration:** Averaging patch features can dilute a rare diagnostic detail, while a class token may preserve a global decision-relevant summary. That is a possible explanation, not a measured mechanistic finding here. The actual ablation shows sensitivity, not the feature responsible for it. Its zero-of-75 bound would be about 3.9%, wider than zero of 150; copying the main table’s bound would be another error.

**Boundary/nonclaim:** do not combine the spider’s ablation 50.7/2.1 with the replication table’s 48.9/2.0 as if they were inconsistent reruns of identical data. They use different capped pools.

**Self-check:** Why does the strong spider aggregation effect not automatically explain the decisive mushroom result?

**Answer:** The mushroom ablations show little DINOv2 change across readouts and no rescue of I-JEPA by max pooling. An effect on one pair must be tested on the pair supporting the inference.

### F42. PCA width contributes to the residual but does not account for the whole gap

**Source locator:** Discussion mechanism (a), main pp. 47–48; S1 §G and Table G, pp. 13–15.

**Passage role:** checks whether a common compression setting handicaps encoders differently.

**Plain-language explanation:** The main protocol keeps 48 principal components. S1 reruns the hard-pair probe with 96, 192, 384 and no projection, holding the cached features and other stated settings fixed. I-JEPA improves substantially. On honey mushroom/Galerina its mean falls from 3.5% at 48 components to 1.8% at 384; on honey/sulphur tuft it falls from 21.3% to 12.0% without projection.

Thus the projection enlarges the baseline’s residual. The source explicitly narrows its attribution instead of claiming pure encoder identity explains everything. Without projection, I-JEPA still has a higher mean than DINOv2 on every tabled hard pair. DINOv2 is comparatively stable across the sweep. However, the width-sweep script does not emit per-seed counts, so a displayed zero at a wider width is not promoted to an observed all-seed zero for a new inferential claim.

**Evidence type:** reported parameter sensitivity on existing cached features. Nonzero means and wider-width parentheses remain descriptive diagnostics.

**Reasoning or worked illustration:** PCA orders directions by variance, not by species discrimination. The top 48 directions from one representation can retain less relevant signal than 48 from another. Increasing width asks whether excluded directions help. A gain demonstrates that compression mattered, but it does not identify the whole causal pipeline. The original investigation also used 64 components and a different rule, so the new width sweep informs attribution indirectly rather than reproducing the original setup exactly.

**Boundary/nonclaim:** a wider I-JEPA diagnostic that falls below the old reference is not a count-based counterexample if it is derived from a nonzero mean. A favorable displayed zero is not enough without the required count audit.

**Self-check:** What two conclusions about PCA can coexist here?

**Answer:** The common projection contributes to I-JEPA’s residual, and removing it does not erase the mean performance gap. Neither conclusion makes every wider-width table parenthesis a valid inferential interval.

### F43. Curated data can mislead both optimism and pessimism

**Source locator:** Discussion mechanism (b), main p. 48; Scope of the result and Limitations, pp. 51–54; Ethics, p. 59.

**Passage role:** connects the sampled benchmark to the real observation regime named by a broad limit claim.

**Plain-language explanation:** Research-grade photographs often display the organism clearly enough for documentation. A strong model can exploit visible whole-organism or contextual features and look nearly perfect. That can encourage unwarranted deployment optimism. At the same time, a weak model’s repeated residual on this stable, curated collection can look like an intrinsic limit rather than a representational failure.

The benchmark therefore cuts both ways. It can be easier than field use for a capable encoder while making a weak encoder’s error appear clean and reproducible. The source does not reject curated benchmarks; it rejects treating their scope as self-evident. Prospective user photographs, damaged specimens, rare stages and deliberately difficult views require data collected for those regimes. A larger sample of the original curated distribution does not automatically represent a different target distribution.

**Evidence type:** interpretation of observational sampling and stated deployment limitations. It is not a new field experiment measuring how much performance degrades in actual foraging.

**Reasoning or worked illustration:** Suppose a hypothetical archive mostly shows mature, unobstructed plants, while a proposed application receives detached leaves. More mature whole-plant photographs tighten estimates for the archive distribution but do not supply the missing detached-leaf evidence. The sample-size question and target-regime question are separate. The same distinction applies when a person selects one especially ambiguous organism rather than taking a random draw from the benchmark.

**Boundary/nonclaim:** near-zero average harmful clearance does not bound the worst specimen. The paper’s selected crops and corruptions are stress tests, not a complete proxy for every field difficulty or the real loss of an error.

**Self-check:** Why is “we tested many more curated images” not a complete answer to a concern about rare, difficult life stages?

**Answer:** It increases precision for the sampled regime but may not change coverage of the relevant rare or selected regime. The missing observation conditions need targeted evidence, not only a larger denominator.

### F44. S1 Proposition 1 is conditional on actual class coincidence

**Source locator:** S1 §A, pp. 1–2; Discussion mechanism (c), main pp. 48–49.

**Passage role:** states the formal constrained-classification limit and identifies the unmeasured antecedent that prevented it from explaining the old residual.

**Plain-language explanation:** Let X be a photograph, Y its edible/poisonous label and φ(X) a fixed representation. The source defines the two class-conditional feature laws P₀ and P₁. A rule chooses clearance or abstention, and may randomize. For a target edible yield α, the best achievable false-eat is a constrained Bayes value. The formal result connects it to thresholding a likelihood ratio, with randomization at ties if needed.

On a region where the relevant class laws genuinely coincide, the features provide no distinguishing information there, and clearance cannot favor edibles over poisons within that coincidence regime. Under that coincidence assumption, the proposition also gives a likelihood ratio of one, posterior equal to the prior, zero conditional label information and zero relevant divergence/Wasserstein separation on the coincident laws. These are consequences of the assumed equality, not empirical estimates established by the failed encoder. The important word is “if.” A failed probe or finite-sample resemblance in one feature space does not measure exact equality of the original image distributions. The paper’s theory can be mathematically useful while its antecedent was never established for the original modality-level attribution.

**Evidence type:** formal statement under defined assumptions. It is not an empirical estimate of a photographic information floor in this dataset.

**Reasoning or worked illustration:** In a hypothetical setting, every member of both classes produces exactly the same observable symbol. A rule using only that symbol can clear all, clear none or randomize, but cannot distinguish the classes from the symbol. If instead an encoder maps two distinguishable photographs to the same symbol, the limitation belongs to that map. These two settings share a feature-level coincidence but differ in what the original observation could support.

**Boundary/nonclaim:** zero estimated divergence in a limited representation is not proof that the photograph’s full class laws coincide. Nor does the existence of a correct Neyman–Pearson argument validate the sampling, labels or deployment scope.

**Self-check:** Which empirical statement was missing from the original reasoning chain?

**Answer:** Evidence that the original photograph, rather than only the selected lossy representation, lacked the discriminating information required at the stated operating constraint.

### F45. Data processing cannot restore information discarded earlier

**Source locator:** S1 §A Proposition 2, p. 1; Discussion mechanism (c), main pp. 48–49.

**Passage role:** explains a valid information bound while showing why it does not privilege the original encoder.

**Plain-language explanation:** The source writes the Markov chain `Y → X → Z` when Z is a transformation of the photograph without an added label-information channel. The data-processing inequality gives `I(Z;Y) ≤ I(X;Y)`. At a fixed yield, the optimal false-eat using Z cannot beat the optimal rule using X itself: `FE*_Z(α) ≥ FE*_X(α)`. This does not say that every representation extracts the same useful information from X.

A weak φ(X) can lose a clue that a different representation ψ(X) retains. Applying more operations only to φ(X) cannot generally recover what it discarded. Returning to the photograph and encoding it differently is not a violation of data processing; it changes the intermediate representation, not the photograph’s own best achievable limit. This distinction is the formal heart of why a correct theorem did not justify the original empirical interpretation.

**Evidence type:** formal reasoning about transformations and feasible rules. It is not a measured mutual-information value for the images or a direct estimate of their Bayes frontier.

**Reasoning or worked illustration:** Any rule using Z=h(X) can be implemented as a rule on X by first computing h. Therefore the set of rules available from X includes those available from Z, which explains why the best X-based error cannot be worse under the same yield constraint. In a hypothetical image with a tiny mark, one compression erases the mark and another retains it. The second can outperform the first while both remain bounded by the information originally in X.

**Boundary/nonclaim:** a generated enlargement may add plausible-looking details without adding true target information. The theorem does not forbid useful preprocessing; it bounds what can be achieved relative to the original observation under its assumptions.

**Self-check:** Does DINOv2 outperforming I-JEPA contradict the data-processing inequality?

**Answer:** No. Both are different transformations of the same photographs. The inequality does not require two lossy feature maps to retain equal task-relevant information.

### F46. An auxiliary observation helps only through a genuinely additional channel

**Source locator:** S1 §A Proposition 3, pp. 1–2; Scope, main pp. 51–52.

**Passage role:** distinguishes a new observation from a more elaborate transformation of the same photograph.

**Plain-language explanation:** An auxiliary measurement W can help only if it adds relevant information conditional on the original photograph, expressed in the source as `I(W;Y|X)>0`. A redundant transform of X plus independent, label-uninformative randomness does not create new target evidence. A conjunction rule that clears only when both the photograph and another channel clear can reduce harmful clearance, but it also sacrifices yield.

Whether it improves the constrained frontier depends on conditional information and the operating point, not merely having two outputs. This formal distinction explains why imagined microscopy inferred from the same image is not automatically equivalent to physically observing a new diagnostic feature. A plausible generated detail can look informative without being a new measurement. The paper’s actual counterexample remains image-based; it does not rely on an additional physical assay.

**Evidence type:** conditional formal statement and a conceptual measurement distinction. No new biological corroboration experiment is reported in this proposition.

**Reasoning or worked illustration:** In a hypothetical independent-channel example, an image rule clears 80% of edibles and 2% of poisons. A second measurement clears 90% of edibles and 10% of poisons. A conjunction would clear 72% of edibles and 0.2% of poisons under the stated conditional independence. That is a lower harmful rate but also lower yield. It is not automatically a solution to the original 80%-yield question. These invented numbers demonstrate the constraint; they are not paper results or practical safety estimates.

**Boundary/nonclaim:** two correlated models on the same image are not automatically two independent observations. Conditional reliability and the data-generating assumptions must be examined. More apparent agreement can reflect shared errors rather than new information.

**Self-check:** Why can “both models agreed” fail to supply the benefit of an auxiliary diagnostic channel?

**Answer:** The models may be redundant transformations of the same observation and share its missing information. Agreement alone does not establish additional conditional information about the label.

### F47. Verification can reproduce the wrong interpretation accurately

**Source locator:** Discussion mechanism (d), main p. 49; Acknowledgments/AI disclosure, pp. 55–57; S1 §C.

**Passage role:** explains why a checking process focused on manufactured gains missed an overclaimed negative.

**Plain-language explanation:** The earlier investigation used adversarial model-agent checks that reproduced results and caught rigged positive claims. Those checks asked whether an apparent gain was real and whether the analysis contained fabricated improvements. They did not adequately ask which strongest plausible baseline could overturn the negative conclusion. An audit can therefore verify that a weak pipeline really fails while leaving untouched the inference that every pipeline must fail.

Correct code and reproducible numbers are necessary for many claims, but their interpretation still depends on the right counterfactual comparison. The source describes this as checking the wrong axis. Its AI disclosure also distinguishes real image/catalog/API inputs from model judgments in the literature audit and synthetic experiments confined to the old failed-method sweep. These distinctions prevent “AI-assisted” from becoming either a blanket invalidation or a substitute for checking evidence.

**Evidence type:** author’s methodological interpretation of the validation history and disclosure of tool roles. It is not independent evidence that any particular panel was free of shared model bias.

**Reasoning or worked illustration:** A hypothetical auditor confirms that ten thermometers all read the same low value after being placed behind the same insulating cover. Reproducing the reading does not establish that the room itself is cold. Removing or challenging the shared bottleneck asks a different question from checking that the instruments agree. The analogy illustrates the verification gap rather than diagnosing the encoders’ physical mechanism.

**Boundary/nonclaim:** the lesson is not that reproduction, theory or adversarial checking is useless. It is that their tasks must include the assumption that carries the negative claim. Separate instances of one model are not independent human expert validation.

**Self-check:** What role instruction would have directly challenged the original modality-level conclusion?

**Answer:** Identify and test the strongest plausible method or representation whose success would collapse the limit claim, while holding the task and operating constraint appropriately fixed.

### F48. The six-check procedure is designed to narrow negatives, not certify impossibility

**Source locator:** S1 §D, pp. 6–7; main Discussion “From mechanism to procedure,” pp. 49–50.

**Passage role:** translates the self-refutation into an actionable research procedure.

**Plain-language explanation:** The procedure asks researchers to name the strongest plausible baseline, rerun across substantially different representations, test the theorem’s antecedent, collect data for the claimed difficult regime, attach count-based bounds, and restate conclusions under the real loss and operating constraint. These are complementary checks. A strong model challenges a common bottleneck; an antecedent audit checks the connection to the formal claim; fresh hard data tests regime transfer; careful counts and losses discipline interpretation.

Passing the list is not a proof that a universal floor exists. The procedure is asymmetric because it is easier to find a scoped counterexample than to exhaust all possible methods and observations. That asymmetry is a feature of the scientific task, not an excuse to select a convenient model or ignore an unfavorable correction family.

**Evidence type:** methodological recommendation supported by the case study. Its general usefulness is proposed; it is not evaluated in a prospective study of every field’s negative claims.

**Reasoning or worked illustration:** In a hypothetical archive-recognition project, first challenge the selected representation before spending months on many downstream classifiers. If no counterexample appears, inspect whether the scan itself lacks the necessary marks and whether the sample includes the damaged regime named in the claim. Then report counts and an application-specific cost boundary. The sequence prioritizes cheap ways to break an overstrong claim without pretending that surviving them proves impossibility.

**Boundary/nonclaim:** no numerical checklist score, automatic approval or field deployment follows from satisfying six items. The required observations and acceptable losses differ across domains. A stronger baseline can narrow a negative without validating a practical application.

**Self-check:** Which statement matches the source: “passing all six proves an information limit” or “failing one may refute or sharply narrow a limit claim”?

**Answer:** The latter. The procedure is a guard against overstrong negatives, not a finite certificate covering every possible image-based rule.

## Scope, corrections and reproducibility

### F49. The cost of an error survives an impressive benchmark

**Source locator:** Scope of the result, main pp. 51–53; Conclusion and Ethics, pp. 54–55 and 59; S1 §D item 6.

**Passage role:** blocks the inference from a refuted limit claim to a deployable biological decision rule.

**Plain-language explanation:** The paper uses lethal identification as a research vehicle precisely because average success can be misleading when one harmful clearance is costly. AUROC can round to 1.00 while the chosen operating threshold still clears poisons. A small false-eat estimate can also have a materially larger upper bound. The source cites an asymmetric-cost example from prior fungi evaluation and asks whether the decision remains acceptable near the bound, not merely at the observed point.

That is a decision-analysis question requiring loss, yield, prevalence and the actual target regime. The study does not choose a universally acceptable harmful rate. It reports that abstention may remain preferable and explicitly states that no operating point is certified safe to eat from. A methodological counterexample removes an overstrong scientific attribution; it does not remove the real-world consequences of an error.

**Evidence type:** source scope statement and cost-aware recommendation. No conducted field trial, ingestion outcome study or validated deployment utility calculation is reported.

**Reasoning or worked illustration:** In a hypothetical equal-sized set of edible and poisonous cases, clearing many edibles can look attractive until a high cost is assigned to each poisonous clearance. The same conditional rates produce different utilities under different costs and encounter mixtures. That is why a single “accuracy” or AUROC figure cannot determine the action. This qualitative illustration makes no numerical recommendation about foraging and supplies no acceptable-risk threshold.

**Boundary/nonclaim:** a more accurate representation does not justify relaxing the source’s abstention conclusion. Nor should a low sampled rate be described as a guarantee for a selected Galerina or Amanita at a difficult life stage. The result is about information available in sampled photographs, not an assurance of personal safety.

**Self-check:** What changes when a study refutes the claim that an encoder’s residual is a universal photographic floor?

**Answer:** The scientific explanation of the residual changes. The acceptable loss, deployment distribution, unknown species and individual-specimen uncertainty still require separate evidence and may still favor abstention.

### F50. S1’s license audit concerns reproducibility and inference separately

**Source locator:** S1 §E and Table F, pp. 7–11; main artifact-selection discussion, pp. 32–33 and 52–53.

**Passage role:** records which artifacts a reader can obtain and use without confusing access with evidentiary admissibility.

**Plain-language explanation:** The source separates a license from an access gate. A model can be downloadable but restricted in use, or permissively licensed but behind an approval process. DINOv2 is recorded as Apache-2.0 and ungated, so the counterexample does not require a manually approved checkpoint. I-JEPA is noncommercial but ungated; MetaCLIP and Web-SSL also have noncommercial terms in the dated audit.

CLIP’s card does not declare a license there, which the source treats as unresolved rather than assuming permission. BioCLIP’s artifacts are MIT and ungated, yet remain excluded by the scientific label-supervision criterion. DINOv3 and other surveyed successors have their own dated access/license limitations and are not run. These facts concern the source’s recorded audit dates. They do not constitute a current legal determination for every use or an automatic reason to accept or reject a model’s scientific evidence.

**Evidence type:** dated documentary audit and methodological interpretation. The source also corrects its initial overgeneralization that all relevant artifacts were permissive.

**Reasoning or worked illustration:** A hypothetical reader may be technically able to download an artifact but unable to use it for a commercial purpose under its terms. Another artifact may be entirely accessible yet trained with the target labels, making it unsuitable for this paper’s narrow inference. These are different questions. A reproduction plan should record both rather than use a single “open” badge that hides what a reader can actually obtain, do and infer.

**Boundary/nonclaim:** this companion grants no rights to weights, images or datasets and changes no license. It does not claim noncommercial terms automatically bar academic checking, nor that a permissive license proves clean evaluation provenance.

**Self-check:** Why can the most permissively licensed models in Table F still be excluded from the main counterexample argument?

**Answer:** The exclusion is about taxonomic target exposure under the study rule. Licensing governs allowed use and access, not whether the resulting score supports the claimed information inference.

### F51. S1 corrections 1–5 teach why artifact-level checks matter

**Source locator:** S1 §F, pp. 11–12; corresponding main statistical, crop, open-set and licensing passages.

**Passage role:** records errors found inside the new analysis rather than presenting the self-correction as a flawless rescue of an old mistake.

**Plain-language explanation:** The first five corrections address undersampled zeros, hand-computed lower bounds, a crop-cache collision, a rounded open-set zero and an overgeneralized license claim. Each has a different remedy. Increasing the test count tightens a zero bound. Recovering integer per-seed counts prevents a rounded mean from masquerading as binomial data.

Checking cache parameters reveals that a file belongs to the wrong crop condition. Reading the actual numerator reveals that a whole-percent zero was 0.17%. Querying each artifact’s terms prevents one favorable license from being assumed for all. The source notes that several errors were caught by instruments or provenance checks rather than rereading alone. Keeping this ledger beside the results prevents the new study’s methodological message from being treated as a claim of personal infallibility.

**Evidence type:** reported correction history with named affected artifacts. It is not a new experimental confirmation that every remaining error has been eliminated.

**Reasoning or worked illustration:** A hypothetical script can produce a mathematically exact interval from an invented rounded count. The interval function can be correct while the input is wrong. Similarly, a cache can be internally readable and externally misidentified. Verification must therefore test inputs, identity and provenance in addition to formula implementation. These examples explain how a technically successful computation can support an invalid interpretation.

**Boundary/nonclaim:** a correction does not automatically overturn the entire conclusion, and preserving corrections does not itself prove the conclusion true. The source states which changes affect numerical detail, scope or evidence status. A teaching guide should follow those distinctions rather than flattening the ledger into either “all fixed” or “all invalid.”

**Self-check:** Which correction would not be caught merely by testing the Clopper–Pearson formula on known integer inputs?

**Answer:** Several: the wrong crop cache, an incorrect license assumption, or a mean mistaken for an observed count. The formula can be correct while artifact identity or statistical input is wrong.

### F52. S1 corrections 6–11 change what the provenance analysis may claim

**Source locator:** S1 §F, pp. 12–13; main Training-catalog overlap and domain-model sections; S1 §I.

**Passage role:** documents changes made while preparing the current version’s overlap and audit evidence.

**Plain-language explanation:** The first overlap analysis hashed a later reacquisition rather than the images actually scored, reporting 18/2,867 pooled matches. The corrected scored-set analysis gives 95/2,775 and zero of 884 recovered on the decisive pair. The first run also used only the newer catalog revision; the corrected version joins both, with identical resulting counts. These are scientific provenance changes, not cosmetic edits.

The source narrows the BioCLIP exclusion to shared source-distribution and label supervision, rather than claiming that shared source proved image overlap everywhere. The audit ledger corrects a known-answer code and its reason, and an overbroad Flickr note is replaced by an adjudication rule. Correction notices on the earlier preprint themselves changed in sequence, including withdrawal and restoration of the exclusion. The current version records the final scope instead of treating every earlier notice as simultaneously valid.

**Evidence type:** reported errata, retained earlier records and corrected analyses. The companion explains the current result and does not promote superseded counts into alternative current findings.

**Reasoning or worked illustration:** A hypothetical model is scored on collection A but audited for overlap using collection B, acquired later from the same species. Even perfect matching of B answers the wrong question about A. This is an identity error upstream of the join algorithm. Fixing the catalog code without fixing the image set would not repair it. Versioned retained outputs let readers see which experiment a number belongs to.

**Boundary/nonclaim:** the corrected 95 matches do not identify how much performance was inflated. The low decisive-pair recovered overlap does not remove the separate label-supervision ground or prove that all training copies are absent.

**Self-check:** Why is it insufficient to say the first overlap run used “the same species”?

**Answer:** The estimand concerns the photographs actually scored. Different photographs from the same species can have different training membership and dates, so species identity does not preserve image identity.

### F53. S1’s script ledger links each result to a reproducible operation

**Source locator:** S1 §J, pp. 21–24; main Data availability, pp. 57–58.

**Passage role:** identifies the code, caches and outputs behind the reported tables and audits.

**Plain-language explanation:** The appendix names separate scripts for ordinary pairs, hard pairs, the eight-encoder extension, plants, other taxa, aggregation, projection width, shuffled labels, held-out calibration and the original operating rule. Full mushroom results require the documented full-image setting; default capped runs produce a different sample size. Per-seed-count scripts replicate the evaluation loops and are checked against archived outputs before their counts support interpretation.

The eight-encoder script was reconstructed from a surviving log and common evaluation code; an original copy was later found and compared. That history matters because a reconstructed script and an original artifact are different provenance states even when their reported outputs match. The point of the ledger is to make each number’s producing operation identifiable, not to imply that one command reproduces every acquisition and rights check.

**Evidence type:** source-reported reproducibility procedure and comparisons on archived caches.

**Reasoning or worked illustration:** A hypothetical evaluator reproduces a capped run with 60 poison images and gets a zero. The paper’s full run uses 150. Identical displayed rates do not mean the bound or experiment matches. Reading the invocation and cache provenance is therefore as important as reading the output. Similarly, a number-to-log match demonstrates traceability, while a clean-room execution with the required inputs demonstrates another layer of reproduction.

**Boundary/nonclaim:** missing raw photographs, unknown licenses or a reconstructed script cannot be wished away by a successful arithmetic test. Conversely, a reconstructed script can be useful if its scope and checks are explicit. The source distinguishes these layers.

**Self-check:** Before claiming a table has reproduced, which four identities should be matched?

**Answer:** The producing script/version, invocation/settings, exact input/caches and intended output/statistic. Matching only a rounded percentage is not enough.

### F54. Platform-dependent nonzero counts remain part of the record

**Source locator:** S1 §J, pp. 22–25; main Acknowledgments numerical-provenance statement, p. 57.

**Passage role:** distinguishes the stable central zero from some nonzero results that move across execution environments.

**Plain-language explanation:** The source reports checks across software and operating-system environments. On a later macOS version, every all-seed zero in the paper reproduces, including the decisive zero of 150. Some nonzero vectors change: thirteen of thirty-one recorded vectors move by at most three specimens in a seed. The original-pair worst seed, woodtuft result and open-set counts therefore differ from the reference-system table in the documented later run.

The out-of-fold count is executed on that later system and includes a reproduced same-pair vector with two one-clearance seeds. These details prevent a reader from treating every nonzero count as bit-exact or calling every discrepancy a new biological result. Floating-point fitting can alter a threshold decision near a boundary even when code and caches are unchanged. The record states which conclusions survive those checks rather than promising universal numerical identity.

**Evidence type:** reported platform sensitivity and retained outputs. It is not a permission to ignore arbitrary mismatches or assume every future platform preserves the same central result.

**Reasoning or worked illustration:** A hypothetical score lies extremely close to a decision threshold. A tiny numerical change can move one specimen from abstain to clear while barely changing the mean score. This explains how low-order computation differences can become integer-count differences. It does not prove that a particular observed discrepancy is harmless; a reproducibility check should locate and report it, as the appendix does.

**Boundary/nonclaim:** the reference-system Table 2 remains the basis of its printed bounds. Later-platform counts must not silently replace only the unfavorable rows while retaining favorable numbers from another environment. The central stable-zero report also needs rechecking in any genuinely new reproduction.

**Self-check:** Why should an education guide report both the reference count and the existence of platform movement?

**Answer:** It preserves what the paper actually estimated and what reproducibility later revealed. Hiding either would falsely imply uniform bit-exactness or a different original experiment.

### F55. Replaying retained hashes is narrower than reacquiring and rehashing photographs

**Source locator:** Data availability, main pp. 57–58; S1 §J, pp. 23–25; records deposit version 4 cited as Zenodo 23569075.

**Passage role:** states what the deposits contain, what the replay reproduces and which inputs remain outside it.

**Plain-language explanation:** The public records deposit contains code and outputs for the catalog measurement and field audit. The additional retained-provenance package supplies ordered query fields, perceptual hashes and a replay route for matching and joins. Replaying those retained inputs can reproduce stored outputs without repeating all network checks or re-hashing the scored photographs.

The source explicitly says that aspect-ratio confirmations are read from deposited results and that photographs themselves are not redistributed. Catalog extracts and GBIF-derived material have source-specific rights; the recovered photographs’ licensing differs from unrestricted public-domain data. The paper records that all 2,775 recovered images come from the iNaturalist research-grade dataset under noncommercial terms, while licenses for other images were unresolved. The educational addition does not copy images or create new permissions.

**Evidence type:** deposit contents, author-reported replay and rights/access boundaries. A stored-hash replay is useful evidence of computational consistency, not an independent fresh image acquisition.

**Reasoning or worked illustration:** A hypothetical checksum verifies that a retained file has not changed. It does not independently show that the file originally came from the claimed camera or website. Replaying a join from preserved identifiers checks the join operation, while reacquiring photographs and repeating visual confirmation checks earlier links. Both can be valuable, but the stronger claim needs the stronger operation. Naming the boundary lets readers plan realistic reproduction rather than discover hidden missing inputs later.

**Boundary/nonclaim:** “available records” does not mean every source artifact may be redistributed freely or that every network-dependent stage can be replayed offline. This guide does not publish private review-access tokens or assert that all underlying rights are resolved.

**Self-check:** What should a successful retained-hash replay be called?

**Answer:** A reproduction of the specified retained-input matching/join outputs. It should not be called a fresh re-hashing of scored photographs or a full end-to-end network reacquisition.

### F56. The limitations and administrative record close the argument honestly

**Source locator:** Limitations, main pp. 53–54; Conclusion and Supporting information, pp. 54–55; Acknowledgments, availability, ethics and declarations, pp. 55–59; References, pp. 59–65; S1 references, pp. 24–25.

**Passage role:** gathers scope, responsibility and access statements rather than leaving the reader with an unqualified headline.

**Plain-language explanation:** The limits ledger names one scoped counterexample, the post hoc original-pair count, an unfixed correction family, dependent and small replication samples, unmeasured DINOv2 image exposure, incomplete recovery, catalog-versus-training-run uncertainty, unreachable re-hosted copies, a small incomplete audit, same-model coding and self-dated plans. The developed units explain each where it changes interpretation. The administrative record adds a responsibility and assistance account: agents produced analyses and drafts; the author made or delegated decisions; audit codes are model judgments; real inputs were not generated or imputed.

Funding, interests, contributions and ethics are declarations, not extra experiments. The bibliography is accounted for as the reference index to the source’s claims, not silently omitted or reclassified as newly verified evidence. This companion teaches the cited claims where they function in the argument without claiming an independent scientific audit of every cited paper.

**Evidence type:** source limitations and author disclosures. Their inclusion supports transparent reading but does not certify that no undisclosed limitation or error exists.

**Reasoning or worked illustration:** A hypothetical project can provide careful code, retained errors and a candid assistance statement while still needing independent reproduction. Transparency makes checking possible; it does not complete every check. Equally, acknowledging AI assistance does not turn real photographs into synthetic data. The source distinguishes measurement inputs from model-coded judgments, and a learner should preserve that separation.

**Boundary/nonclaim:** the source’s disclosures do not establish food-safety assurance or independent scientific validity. This explanation is not a replacement for the exact preprint.

**Self-check:** What is the most defensible closing sentence for a reader who understands the full paper?

**Answer:** A stronger representation provides a bounded, qualified counterexample to a broad photographic-floor attribution, while deployment safety, every-specimen reliability and several provenance and inferential uncertainties remain unresolved.

## Worked reading of the supplementary coding record

### F57. Following three adjudications down to the scored model and table

**Source locator:** S1 §I per-paper coding table and later-check explanations, pp. 16–21; main Results, pp. 44–45.

**Passage role:** makes the documentary audit concrete by showing why a paper’s title or one table is insufficient for its paper-level code.

**Plain-language explanation:** The later model check disagrees with three final codes. For DALIP, it looks only at BioCLIP results and gives unknown. The source points to a table that also scores ArborCLIP on the Rare Species benchmark, making the paper positive under the manual’s rule. For COT-AD, a negative reading based on the authors’ own images misses external cotton-leaf test sets of untraceable lineage, so the final code is unknown. For Prompt-CAM, a positive reading cites iNat21 and Fish-Vista evaluations that use a DINO backbone rather than the BioCLIP backbone relevant to the audit; its BioCLIP evaluations have different provenance, leaving unknown. These are distinctions between model, dataset, table and reporting unit. The final primary count remains eight positives, eleven negatives and five unknowns. The appendix supplies the per-paper record so the reader can inspect the basis rather than accept agreement statistics alone.

**Evidence type:** source-reported documentary adjudications. They are not new measured image-overlap results and do not establish that a model’s accuracy was inflated.

**Reasoning or worked illustration:** A hypothetical article can use dataset A with model X and dataset B with model Y. Combining “model X is in the paper” with “dataset B is in the paper” invents an evaluated pair that may never occur. Conversely, reading only the first table can miss a later relevant pair. Accurate coding must follow the actual scored combination, its source lineage and the manual’s definition of a qualifying check. This is why article-wide keyword matching is insufficient.

**Boundary/nonclaim:** the later check is nonblind and model-generated. Correcting three cases does not create a human gold standard or prove that every remaining code is correct. The whole per-paper table remains an inspectable record, not an invitation to generalize beyond the frame.

**Self-check:** What join keys should a careful reader preserve when checking one positive audit code?

**Answer:** The paper, exact model/backbone, scored test dataset, relevant table or result, source lineage and qualifying-check evidence. Losing any of them can turn a real evaluation into a falsely inferred one.

## Implementation and correction ledger for source tracing

This ledger is a compact reference beside the developed lessons. Settings and formulas are source definitions; counts are reported source outputs. Nothing here asserts a fresh rerun. A table cell can be fully accounted for by a lesson about its row, statistic and limitation without inventing a separate interpretation for every decimal.

| Source item | Exact setting or reported structure | Teaching route and boundary |
|---|---|---|
| Main task/S1 A | X = one RGB photograph; Y = 0 edible, 1 poisonous; φ fixed features; `FE=Pr(clear given Y=1)`, `Yield=Pr(clear given Y=0)` | F04–F05, F44–F46. Conditional rates do not equal risk among clearances. |
| Original rule | One split, 7/81; lower 95% bound 4.13%; reconstructed yield 68.5–78.7%; 64-component I-JEPA | F06, F21. The old approximate 4.9% ± 2.5% summary has no counts for an exact interval. |
| Main probe | Ten 70/30 splits; PCA 48 fitted on training; 12-member 128/32-unit MLP ensemble, Adam 1e−3, batch 256, 400 steps, per-member bootstrap | F08–F10. Frozen encoder does not mean unfitted probe. |
| Yield threshold | Test-edible percentile giving 80% yield, no held-out poison labels used | F09, F25. Empirical frontier, not guaranteed prospective yield. |
| Reporting | Table mean over dependent seeds; diagnostic count = round(mean rate × n) | F10–F12. A derived integer is not necessarily an observed count. |
| Exact bounds | One-sided zero upper = 1 − α^(1/n); floor lower from 7/81; family adjustment α = .05/m | F06, F11–F12, F20–F21. Per-cell versus family/sampling assumptions remain visible. |
| Main Table 1/S1 Table H | Five ordinary pairs; n = 720 pair-scorings/570 unique specimens; no pooled observed zero | F18. Per-pair means and audits do not make the pooled statistic independent. |
| Novel set 1 | 906 images × 5 classifiers × 10 seeds = 45,300 scorings; DINO about .48%, rounded 0% in the old format | F19. Distinct from the 1,151-image set. |
| Main Table 2 | Decisive 0/150 per seed; original-pair worst 2/150 reference; other zero-of-120 cells; open-set worst 4/1,151 | F20–F24. Rows have different roles, not equal refutation status. |
| Main Table 3 | Eight encoders × five hard pairs; first three n = 150, last two n = 120 | F22, F26–F30. Supervision/source categories do not isolate objective causally. |
| Crops / degradation | 55% ordinary versus 50% hard crop; cache collision corrected; DINO-only degradation; not every regime on the decisive pair | F23, F51. Engineered perturbations are not prospective field data. |
| Calibration/null | Shuffled AUROC .50/false-eat 79.2%; separate calibration yield 61.7–90.0%, mean 79.2% | F25. Similar numeric means summarize different controls. |
| Main Table 4 | Both BioCLIP models AUROC 1.00; BioCLIP 2.5 all hard-pair zeros | F28. Excellent performance is not a reason to hide an exclusion criterion. |
| Catalog recovery | pHash 64 bits, Hamming ≤6, unique assignment, aspect-ratio confirmation; two pinned revisions | F13–F14, F29–F31. Unrecovered and re-hosted images remain outside the exact join. |
| Main Table 5 | 3,600 scored; 2,775 recovered; 95 matches; control 95/868; decisive 0/884; earthball 39/400 recovered | F29. Worst and bounded-worst cases are missingness sensitivities. |
| Audit rules | Common lower >25%; rare upper <10%; positive = at least one same-source evaluation without a qualifying check | F15–F17, F39–F40. Category and estimand are not measured leakage. |
| Main Table 7 | 8 positive, 11 negative, 5 unknown, N = 24; primary indeterminate; 13/24 unknown-positive bound | F39. No human coding or field-wide prevalence inference. |
| Main Table 8 | Capped 250/species, n = 75; I-JEPA mean/max; DINO CLS/mean/max | F41. Different pool from full tables; zero-of-75 upper about 3.9%. |
| S1 Table G | K = 48, 96, 192, 384, raw across five hard pairs; no sweep per-seed counts | F42. Wider displayed zeros remain descriptive without audited counts. |
| S1 Table A | Three Apiaceae pairs, n = 90 | F34. Nonzero mean parentheses are diagnostics. |
| S1 Table B | Three coral-snake pairs, n = 90; two I-JEPA harmless-pool exceptions | F35. Dangerous n alone does not establish perfectly paired data. |
| S1 Table C | Three recluse pairs, n = 90 | F36, F41. Human misidentification and image information differ. |
| S1 Table D | Four frog pairs, n = 22–90; two Müllerian; some same-occurrence media | F37. Small n and event definition constrain interpretation. |
| S1 Table E | Three berry pairs, n = 90; one easy I-JEPA pair | F38. Representation dependence is task-specific. |
| S1 F corrections | Eleven items: undersampling, intervals, crop, rounding, license, label/image grounds, wrong images, catalog revision, known answers, Flickr, notices | F51–F52. Retained earlier outputs are not simultaneous current results. |
| S1 I | Per-paper code table and self-dated/amended timeline; later nonblind model check | F15–F17, F39–F40. No human-code provenance is invented. |
| S1 J | Named FULL/CROP invocations, original/reconstructed script, per-seed audits, platforms, retained-input replay | F53–F55. A log check, cache rerun and fresh acquisition are different checks. |

## Glossary

- **Abstention:** withholding a clearance rather than forcing a class decision. It changes yield and may be the appropriate practical action.
- **Admissible/label-free:** the source’s no-taxonomic-supervision category / its stricter no-text-or-label-channel category. Neither means proved image-disjoint.
- **AUROC:** ranking quality across thresholds; not the harmful-clearance rate at a particular yield.
- **Bayes frontier/floor:** the best achievable error under a defined observation, distribution and constraint. It is not automatically measured by a failed fitted model.
- **Bonferroni family:** the set of comparisons counted when allowing for multiple opportunities to find a favorable result.
- **Clopper–Pearson:** a binomial confidence construction based on integer counts and sampling assumptions.
- **Conditional rate:** a probability within a specified group, such as poisonous specimens, rather than among all outputs.
- **Counterexample:** an included case or rule that violates a universal assertion at the asserted scope and operating condition.
- **Data processing:** transforming an observation cannot improve on the information available in that original observation under the stated assumptions.
- **Diagnostic parenthesis:** the source’s descriptive scale calculation from a rounded seed mean, distinct from an exact interval for an observed count.
- **Distribution shift:** a change between the sampled observations and the regime where a conclusion or model is applied.
- **Embedding/representation:** numerical features produced from an image by an encoder; features can discard information.
- **False-eat/false-safe:** harmful specimens cleared divided by harmful specimens tested, with event definitions appropriate to the task.
- **Frozen:** encoder weights are not fine-tuned for the new task; the projection and probe can still be fitted.
- **Information limit:** a claim about all suitable rules using the specified observation, stronger than a statement about tried methods.
- **Likelihood ratio:** relative probability/density of an observation under the two classes; the formal constrained rule uses it.
- **Müllerian mimicry:** resemblance between defended species; the corresponding rows test species separability rather than safe-versus-dangerous clearance.
- **Occurrence:** the biological observation record that can supply one or more photographs. Several media may share one occurrence.
- **Out-of-fold:** each observation is tested by a rule fitted without it, while fitted folds can share other training data.
- **PCA:** principal component compression based on variation, not guaranteed task relevance.
- **Perceptual hash:** a compact image signature used here for candidate matching; it is not a substitute for all provenance checks.
- **Posterior/prior:** label probability after observing features / before those features. Coincident class laws cannot supply the missing distinction.
- **Precommitment/post hoc:** fixed before the relevant analysis or evidence / designed after earlier results are known. Before execution alone does not settle the distinction.
- **Probe:** a downstream learned predictor from frozen features.
- **Seed:** a random-number initialization controlling a repeat; it is not an independent collection of organisms.
- **Snapshot/catalog:** a dated declared collection or membership record; it is not direct observation of every training operation.
- **Yield:** the fraction of edible/harmless specimens cleared under the defined rule.

## Twelve answered transfer questions

Try each self-study question before reading its answer.

1. **A medical-image team says the modality cannot detect a condition after five models fail. What first distinguishes model failure from observation failure?** Test a strong, substantially different representation on the same patient-separated data and clinically relevant operating constraint, with provenance and count-based uncertainty. A successful scoped counterexample challenges the broad attribution; failure of a few more models does not prove impossibility.
2. **A classifier rejects every case and has no harmful clearances. Has it solved the constrained task?** No. Its yield is zero. Compare error at the required yield and do not hide abstention behind an attractive harmful rate.
3. **A zero count is based on 60 observations rather than 150. What changes?** Its one-sided 95% upper bound is about 4.87% rather than 1.98%. Identical observed proportions can leave different uncertainty.
4. **Ten random splits each test 150 overlapping images. Can n be reported as 1,500 independent specimens?** No. Repeated scorings are dependent and images recur. Preserve unique-specimen counts and the exact analysis unit.
5. **A candidate upper bound is 4.14% and the old lower bound 4.13%. What does one-decimal equality hide?** The upper bound is still slightly higher. The source’s separation criterion is not met.
6. **A zero-of-150 cell survives a six-model correction but not a thirty-cell correction. What should be taught?** The same observed count has family-dependent adjusted inference, and the family was not fixed in advance. Report the sensitivity rather than selecting the convenient family as though prespecified.
7. **A model declares the same image source as the test set. Has actual overlap been measured?** No. A declaration identifies a possible lineage. Exact membership needs a suitable identifier/hash join, and even that has coverage and catalog-versus-training-run limitations.
8. **A test image was created after a frozen snapshot. An older image is also absent from a join. Are the two absences justified in the same way?** No. The date can exclude the newer image from that snapshot; the older one’s absence relies on join coverage and identifiers. Early creation alone neither proves nor disproves membership.
9. **An audit’s primary result is 8/24, CI 15.6–55.3%, under a common rule requiring lower >25%. What is the category?** Indeterminate. The point estimate 33.3% does not satisfy the lower-bound condition, and the event is a reporting-practice code rather than measured contamination.
10. **Two model coders agree on nearly every paper. Is independent human reliability established?** No. Shared model and instructions can share errors. Report actual coder identities, adjudication, unknowns and whether a later check was blind.
11. **A replay reproduces stored hash-join outputs. Has the image acquisition reproduced?** Not necessarily. Reacquiring images, computing fresh hashes and repeating network confirmations are earlier links requiring separate inputs and checks.
12. **A stronger image model overturns a broad information-floor claim. What remains before deployment?** A defensible target population, prospective data, acceptable loss and yield, unknown-species handling, individual-case limitations and an operational validation plan. In this paper, real foraging still calls for abstention under uncertainty.

## Exact source

Exact manuscript: [Zenodo record 23578222, v2](https://doi.org/10.5281/zenodo.23578222), `Iben_2026_A_False_Floor_preprint_v2.pdf`, SHA-256 `ca83ef3cfcedeea504ad20281787b07d7dea04f797c450e9f356d9729bf2a761`. Relevant supplement on the same record: `Iben_2026_A_False_Floor_S1_Appendix.pdf`, SHA-256 `52d9fb7630123d435747e105f7e520c0281ae691ddb4a4ae21fe89f12c74410d`. The [version 4 records deposit](https://doi.org/10.5281/zenodo.23569075) is a distinct reproducibility record, not the manuscript.

[COMPLETE_COVERAGE.md](COMPLETE_COVERAGE.md) maps every substantive main/S1 section, table, figure, proposition and limitation, and identifies references and administrative matter. The [concise guide](STUDY_GUIDE.md) provides a shorter reading route.
