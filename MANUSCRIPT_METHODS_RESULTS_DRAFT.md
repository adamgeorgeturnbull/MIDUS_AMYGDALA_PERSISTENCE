# Corrected Methods and Results Draft

This is replacement prose drafted from the final corrected M3 outputs and the final MR1 Box outputs copied on September 7, 2026. It intentionally omits implementation details, filenames, run logs, and other material that would not ordinarily appear in a scientific manuscript. Table 1, figure captions, the Abstract, and the Discussion should be revised separately after this prose is approved.

## Methods

### Design

This study used a cross-sectional observational design to test preregistered hypotheses replicating associations between amygdala neural persistence and daily-life affect and extending these hypotheses to vmPFC–amygdala circuitry. The MIDUS 3 (M3) hypotheses and analysis plan were preregistered on the Open Science Framework (https://doi.org/10.17605/OSF.IO/85H2D) before analysis. Deviations from the preregistered plan are summarized in Table 1. A subsequent, analysis-plan-constrained evaluation in MIDUS Refresher 1 (MR1) examined the corresponding persistence, age, and functional-connectivity associations in an independent cohort.

### Participants

The primary sample was drawn from the M3 daily diary and neuroscience projects. The daily diary sample comprised 1,174 participants (mean age = 67.6 years, SD = 10.3, range = 47–94; 57.3% female). Of these, 137 also had age data from the neuroscience visit. The conservative fMRI sample comprised 127 participants who met all imaging quality-control criteria (mean age = 65.7 years, SD = 9.5, range = 48–95; 59.8% female). This sample was 75.6% White, 18.9% Black, 0.8% Native American, 1.6% Asian, and 3.1% other race. The primary diary-plus-fMRI sample comprised 81 participants (mean age = 67.2 years, SD = 9.8, range = 48–95; 61.7% female), of whom 80 had complete emotion-regulation questionnaire data for moderation analyses. PANAS sensitivity analyses used the full conservative fMRI sample (N = 127), irrespective of diary participation.

The MR1 validation sample contained 123 participants with reproduced imaging measures. Of these, 115 met the MR1 conservative quality-control criteria (mean age = 48.4 years, SD = 11.9, range = 26–76; 53.9% female), and 48 had both daily diary data and conservative-quality imaging data (mean age = 46.9 years, SD = 9.5, range = 29–70; 54.2% female). The conservative MR1 sample was 64.3% White, 28.7% Black, 1.7% Native American, 0.9% Asian, and 4.3% other race.

### Behavioral Preprocessing

**Daily diary affect.** Daily affect was assessed for approximately eight consecutive days. The 27 affect items were rated on a 1–5 scale. Positive affect (PA) was calculated as the mean of 13 positive-mood items, and negative affect (NA) as the mean of 14 negative-mood items. Days with missing responses on any affect item were excluded, and participant-level PA and NA scores were calculated by averaging across complete diary days. Because NA was positively skewed, a secondary log-transformed score was calculated as log(NA + c), where c was half the minimum nonzero NA value. Raw NA was the primary negative-affect outcome; log-transformed NA was treated as a robustness outcome. Corresponding MR1 diary items were scored using the same procedure.

**PANAS.** Positive and negative affect assessed with the PANAS at the neuroscience visit were examined as convergent-validity sensitivity outcomes. Values coded as 8, 98, or 99 were treated as missing before distributional summaries or transformations were calculated. PANAS negative affect was also log transformed using the offset procedure described above.

**Emotion regulation and demographic variables.** Reappraisal and suppression were assessed with the Emotion Regulation Questionnaire at the neuroscience visit. Age at the diary wave was used for diary-only age analyses, whereas age at the neuroscience visit was used in analyses involving neuroscience measures. Biological sex retained the original MIDUS coding (1 = male, 2 = female). Race was represented by five dummy variables with White as the reference category. The interval between the diary and neuroscience visits was calculated in months.

### fMRI Task and Acquisition

Participants completed three runs of an emotional image-viewing task. Each run included 10 negative, 10 neutral, and 10 positive images, followed by neutral face probes. Images were presented for 4 s, followed by a 2-s fixation interval and a 0.5-s face probe. Participants categorized the gender of each face. Negative and positive images were matched on normative arousal, and images within each valence category were matched on luminosity, visual complexity, and social content.

M3 data were acquired on a 3T GE MR750 scanner using a 32-channel head coil. Each of the three functional runs contained 231 volumes (TR = 2,000 ms; TE = 20 ms; flip angle = 60°; field of view = 220 mm; matrix = 96 × 64; 40 interleaved sagittal slices, 3-mm thickness with a 1-mm gap). A T1-weighted BRAVO anatomical image was also acquired (TR = 8.2 ms; TE = 3.2 ms; flip angle = 12°; field of view = 256 mm; matrix = 256 × 256; 160 axial slices; inversion time = 450 ms).

### fMRI Preprocessing and Quality Control

For M3, the analyses reported here used a corrected preprocessing run initiated from untouched raw BOLD images. Because the raw images had a nonstandard spatial-axis layout, images were reoriented and the axes temporarily exchanged before slice-timing correction. Slice-timing correction was performed in FSL 6.0.7.10 using the 1-based interleaved acquisition order derived from the imaging metadata; the axes and original geometry were then restored. Structural and functional data were processed with fMRIPrep 24.1.0, with fMRIPrep slice-timing correction disabled so that slice-timing correction was applied exactly once. Preprocessing included anatomical reconstruction, motion correction, functional-to-anatomical registration, and normalization to MNI152NLin2009cAsym space. Framewise displacement (FD) was derived from the resulting fMRIPrep confound time series.

The conservative M3 sample required all three functional runs to pass visual quality control, mean FD across task runs to be less than 0.5 mm, and all six directional cross-run pairs needed for the primary left-amygdala persistence measure to be available. Visual ratings were evaluated separately from quantitative FD. MR1 imaging data were reproduced using fMRIPrep 24.1.0 with standard BIDS slice-timing handling. Historical visual quality-control ratings were retained, whereas FD was recalculated from the reproduced confound time series. The conservative MR1 sample additionally required all three task runs to contain exactly 231 volumes.

### fMRI Analysis

**First-level general linear model.** A first-level general linear model was estimated separately for each run using nilearn. Six task regressors modeled negative, neutral, and positive images and the neutral face probes that followed images in each of those three valence conditions. Thus, the face regressors indexed the valence of the preceding image rather than the valence of the face itself. Each regressor was convolved with the Glover hemodynamic response function. Models included a 24-parameter motion model, cosine drift terms with a 1/128-Hz high-pass cutoff, and an AR(1) noise model. The first four volumes and corresponding confound rows were discarded.

**Amygdala neural persistence.** Left and right amygdala regions were defined from the Harvard–Oxford atlas at a 50% probability threshold. For each hemisphere, persistence was calculated as the spatial Pearson correlation between the voxelwise beta pattern for negative images in run *i* and the pattern for neutral faces following negative images in run *j*, where *i* ≠ *j*. Correlations were transformed to Fisher *z* values and averaged across the six directional cross-run pairs. Left-amygdala negative persistence was the primary measure; right-amygdala and positive-image persistence were examined in sensitivity analyses.

**Task-based functional connectivity.** Amygdala–vmPFC functional connectivity was estimated with a least-squares-separate beta-series procedure. For each image trial, the trial of interest was modeled separately, other trials of the same valence were combined in a second regressor, and trials from each remaining condition were combined into condition-specific regressors. Each image event was modeled for 6 s, encompassing the 4-s image and subsequent 2-s fixation interval; the following face was not included. Trial-specific beta estimates were extracted from left and right amygdala and from 10-mm-radius anterior (MNI coordinates: −2, 46, −10) and posterior (0, 26, −12) vmPFC spheres. Beta-series correlations were calculated separately for each connection, condition, and run, transformed to Fisher *z*, and averaged across runs. Negative-minus-neutral connectivity was the primary contrast. Analyses focused on left-amygdala connections, with right-amygdala, positive-minus-neutral, and negative-condition-only measures examined as sensitivity analyses.

### Statistical Analyses

For M3, linear mixed-effects models with a random intercept for family were the primary adjusted analyses. Paired twins shared a family grouping identifier; participants without an observed co-twin were assigned individual identifiers. Models were estimated with restricted maximum likelihood. Supplementary analyses included zero-order Pearson correlations and ordinary least-squares (OLS) regression; OLS models included twin-pair indicators instead of the family random intercept.

Adjusted models included age at the neuroscience visit, biological sex, and race dummy variables. Models with diary outcomes additionally included the interval between the diary and neuroscience visits and the number of complete diary days. Age was omitted from the covariate set when it was the predictor. Prespecified directional hypotheses were evaluated with one-tailed tests, calculated from the corresponding two-tailed probability according to the observed direction. Raw NA was the primary negative-affect outcome, with log-transformed NA used to assess robustness. Right-hemisphere and other specificity analyses used two-tailed tests unless an explicit directional prediction had been specified. Exploratory moderation tests were two-tailed. Predictor and moderator variables were mean centered before interaction terms were calculated. Sensitivity and exploratory analyses were reported with nominal *p* values without multiplicity adjustment and were not used to redefine the primary hypotheses.

For MR1, corresponding participant-level OLS models were used because the available analytic inputs contained no usable family, household, sibling, or twin-pair grouping identifier. Models included the same available covariates and directional tests as the corresponding M3 analyses. The conservative MR1 analysis was primary, and models without conservative QC restriction were retained as secondary analyses. Direct results were classified as supported when significant in the predicted direction, directionally consistent when nonsignificant but in the predicted direction, and not supported when the estimate was in the opposite direction. MR1 sensitivity analyses were interpreted as robustness or specificity checks and did not alter the direct-replication classification.

## Results

### Convergence of Affect Measures

Daily diary and PANAS affect scores were positively associated in the 136 participants with both measures: diary PA with PANAS positive affect, diary NA with PANAS negative affect, and log-transformed diary NA with log-transformed PANAS negative affect (all *p*s < .001). These associations support convergence while also indicating substantial measure-specific variance.

### Task-Related Activation and Functional Connectivity

In the conservative M3 fMRI sample, left-amygdala activation was greater than zero for negative, *t*(126) = 5.63, *p* < .001, and positive images, *t*(126) = 4.99, *p* < .001, but not for neutral images, *t*(126) = 1.82, *p* = .070. Right-amygdala activation was positive in all three conditions (all *t*s ≥ 5.30, all *p*s < .001). Both amygdalae responded more strongly to negative than neutral images (left: *t*(126) = 3.77, *p* < .001; right: *t*(126) = 3.17, *p* = .002), whereas negative and positive responses did not differ significantly (both *p*s ≥ .074).

Anterior vmPFC showed significant deactivation during negative, *t*(126) = −3.54, *p* < .001, and neutral images, *t*(126) = −2.99, *p* = .003, but not positive images, *t*(126) = −1.28, *p* = .204. Deactivation was greater during negative than positive images, *t*(126) = −2.98, *p* = .003. Posterior vmPFC activation did not differ from zero or between conditions (smallest *p* = .114).

All 12 amygdala–vmPFC connection-by-condition estimates were positive (all *p*s ≤ .003), indicating reliable task-related coupling. However, no connection differed significantly between negative and neutral or between negative and positive conditions (all *p*s ≥ .097). Across activation, persistence, and beta-series connectivity measures, anterior and posterior vmPFC estimates showed moderate-to-strong correspondence (all *p*s < .001), while retaining substantial nonshared variance.

### Age and Affect

In the full diary sample, older age was associated with higher PA (β = .010, *z* = 4.91, *p* < .001), lower raw NA (β = −.00285, *z* = −4.07, *p* < .001), and lower log-transformed NA (β = −.0222, *z* = −5.28, *p* < .001) in the primary adjusted models (N = 1,168 after covariate-complete-case restriction). In the smaller diary-plus-neuroscience sample (N = 137), age was not significantly associated with PA (*z* = 1.23, *p* = .110) or raw NA (*z* = −.82, *p* = .205), but was associated with lower log-transformed NA (*z* = −2.22, *p* = .013). In the PANAS sample (N = 230), age was not associated with positive affect (*p* = .406) or raw negative affect (*p* = .054), but was associated with lower log-transformed negative affect (β = −.00284, *z* = −1.98, *p* = .024).

### Amygdala Persistence and Affect

In the primary diary-plus-fMRI sample (N = 81), greater left-amygdala negative persistence was associated with higher raw NA in the primary mixed-effects model (β = .278, SE = .136, *z* = 2.05, one-tailed *p* = .020). Associations with PA (β = −.700, SE = .476, *z* = −1.47, *p* = .071) and log-transformed NA (β = 1.950, SE = 1.273, *z* = 1.53, *p* = .063) were in the predicted directions but were not significant.

In the 54-person subsample with no participant overlap with the original MIDUS 2 study, the mixed-effects association remained significant for raw NA (β = .314, *z* = 1.95, *p* = .025), but not for PA or log-transformed NA (both *p*s ≥ .143).

Right-amygdala negative persistence did not significantly predict diary PA or NA in the primary M3 sensitivity models (two-tailed *p*s ≥ .084). Left-amygdala persistence also did not significantly predict PANAS affect in the full conservative fMRI sample (one-tailed *p*s ≥ .088). Positive persistence, vmPFC persistence, and regional activation measures were not significantly associated with diary affect in the corresponding two-tailed sensitivity analyses (all *p*s ≥ .084).

### Age and Amygdala Persistence

Older age was associated with lower left-amygdala negative persistence in the primary mixed-effects model (β = −.00260, SE = .00124, *z* = −2.09, one-tailed *p* = .018; N = 127). Associations between age and right-amygdala negative persistence, bilateral positive persistence, and anterior or posterior vmPFC persistence were not significant in two-tailed sensitivity analyses (all *p*s ≥ .159).

### Amygdala–vmPFC Functional Connectivity and Affect

In the primary diary-plus-fMRI sample (N = 81), greater left-amygdala–anterior vmPFC connectivity for negative relative to neutral images was associated with lower raw NA (β = −.106, SE = .060, *z* = −1.76, one-tailed *p* = .039) and lower log-transformed NA (β = −1.063, SE = .485, *z* = −2.19, one-tailed *p* = .014) in the primary mixed-effects models. It was not associated with PA (β = −.018, *z* = −.08, *p* = .532).

Posterior vmPFC connectivity did not significantly predict any outcome in the prespecified direction (all one-tailed *p*s ≥ .132). Its mixed-effects association with PA was negative (β = −.353, *z* = −2.33), opposite the predicted positive direction; accordingly, the prespecified one-tailed *p* was .990 (two-tailed *p* = .020). Posterior connectivity was not significantly associated with either raw or log-transformed NA (one-tailed *p*s ≥ .132).

In PANAS sensitivity analyses (N = 127), both anterior and posterior left-amygdala–vmPFC connectivity were associated with lower PANAS negative affect in the primary models (raw: βs = −.251 and −.249, one-tailed *p*s = .011; log transformed: βs = −.106 and −.099, *p*s = .014 and .020). Neither connection predicted PANAS positive affect (one-tailed *p*s ≥ .329). Right-amygdala and positive-minus-neutral connectivity did not significantly predict diary affect (two-tailed *p*s ≥ .353). In an exploratory analysis of relative anterior versus posterior connectivity, the left anterior-minus-posterior index predicted higher PA in the mixed-effects model (β = .471, *z* = 2.51, one-tailed *p* = .006).

### Functional Connectivity and Amygdala Persistence

Neither left anterior nor left posterior amygdala–vmPFC connectivity for negative relative to neutral images predicted left-amygdala negative persistence in the primary mixed-effects models (anterior: β = −.018, *z* = −.53, one-tailed *p* = .299; posterior: β = .011, *z* = .32, *p* = .626; N = 127). With both *p*s ≥ .299, these findings provided no evidence that individual differences in the connectivity contrast were associated with amygdala persistence.

### Assessment of Head Motion

Neither mean FD across runs nor mean FD during negative-condition trials was associated with left-amygdala persistence or left-amygdala–vmPFC connectivity in mixed-effects models (all two-tailed *p*s ≥ .129). In motion-adjusted follow-ups, all significant primary mixed-effects findings remained significant after controlling separately for each motion measure: persistence with raw NA (*p*s = .038 and .026), age with persistence (*p*s = .019 and .019), and anterior connectivity with raw NA (*p*s = .048 and .040) and log-transformed NA (*p*s = .018 and .015). These results reduce concern that those tested associations were attributable to head motion.

### Emotion Regulation and Affect

Reappraisal and suppression were not significantly associated (*p* = .638). Neither strategy significantly predicted diary PA, raw NA, or log-transformed NA in the primary adjusted models (all *p*s ≥ .065). In contrast, reappraisal predicted higher PANAS positive affect (β = .178, *z* = 4.16, *p* < .001), whereas suppression predicted lower PANAS positive affect (β = −.102, *z* = −2.63, *p* = .009) and higher PANAS negative affect (raw: β = .059, *z* = 2.51, *p* = .012; log transformed: β = .027, *z* = 2.48, *p* = .013). Age was not significantly associated with either strategy (both *p*s ≥ .204).

### Exploratory Moderation by Emotion Regulation

All moderation analyses were exploratory and used nominal two-tailed tests. In the primary mixed-effects models (N = 80), reappraisal moderated the association between left-amygdala negative persistence and PA (βinteraction = 1.000, SE = .496, *p* = .044), and suppression moderated the same persistence–PA association (βinteraction = .586, SE = .151, *p* < .001). No persistence interaction significantly predicted either raw or log-transformed NA (all *p*s ≥ .120).

For connectivity, suppression moderated the association between left-amygdala–posterior vmPFC connectivity and raw diary NA in the primary mixed-effects model (βinteraction = .097, SE = .049, *p* = .047). No other diary FC interaction was significant (all *p*s ≥ .151). In the PANAS sensitivity analysis, suppression moderated the association between posterior connectivity and negative affect in the mixed-effects models (raw: βinteraction = −.246, *p* = .014; log transformed: βinteraction = −.098, *p* = .026). Given the number of exploratory tests, these moderation results require replication.

### Replication in MIDUS Refresher 1

None of the four direct MR1 tests reached the prespecified significance threshold (all one-tailed *p*s ≥ .102). In the conservative diary-plus-fMRI sample (N = 48), left-amygdala negative persistence was negatively associated with PA (β = −.232, SE = .582, *t*(41) = −.40, one-tailed *p* = .346) and positively associated with raw NA (β = .282, SE = .218, *t*(41) = 1.29, one-tailed *p* = .102), matching the predicted directions but not reaching significance. The log-NA estimate was also positive and nonsignificant (β = 1.352, *t*(41) = 1.06, *p* = .148). Age was positively rather than negatively associated with left-amygdala persistence in the conservative fMRI sample (N = 115; β = .00134, SE = .00147, *t*(108) = .91, one-tailed *p* = .818). Left-amygdala–anterior vmPFC connectivity was negatively but nonsignificantly associated with raw NA (N = 48; β = −.078, SE = .100, *t*(41) = −.78, one-tailed *p* = .220). Thus, the persistence–PA, persistence–NA, and anterior-connectivity–NA results were directionally consistent with M3, whereas the age–persistence result was not supported.

Two predefined MR1 sensitivity regressions were nominally significant (two-tailed *p*s = .041–.046). Right-amygdala negative persistence predicted lower PA (β = −1.302, SE = .633, *t*(41) = −2.06, two-tailed *p* = .046), and right-amygdala–anterior vmPFC connectivity predicted higher NA (β = .240, SE = .114, *t*(41) = 2.11, two-tailed *p* = .041). The latter effect was opposite the direction specified for the left-anterior direct target. Neither association was significant as a zero-order correlation (*p*s = .075 and .154), and no other predefined MR1 sensitivity regression was significant (all *p*s ≥ .070). Because these were nominal findings within broader sensitivity families, they did not alter the direct-replication classification.
