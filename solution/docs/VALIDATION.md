# Validation scope

The model was fitted on seven external DFL fixtures. Whole-match excluded-fit provider-label spellF1 averages were 0.67483 clean, 0.55523 moderate, 0.48437 severe and 0.52913 contextual. These fixtures were examined during development; fold exclusion does not make adaptive architecture research an untouched final test. Longer-possession conventions change rankings.

Metrica Game 3 used a candidate frozen before downloading that fixture. Partial pass/dead references cover 46.99% of frames. Moderate newly missing pass accuracy increased 84.13% to 87.00%, with a paired conditional 95% improvement interval of +1.40 to +4.27 percentage points. Overall moderate labelled-frame accuracy was 88.86%. Ghost-exposed partial accuracy regressed 82.49% to 81.93%; native overall improvement had an interval spanning zero. Bootstrap units were labelled event intervals, not independent matches. No complete Game 3 possession-count or spellF1 claim is made.

The development suite passed 143 tests before consolidation. The active delivery suite passed 70 tests after consolidation and retains model loading, fusion, decoder, feature, reliability and input/output checks. Historical tests and training/validation tools are preserved in Git history and the local snapshot.

The runtime passed external replay checks. Two independent challenge-input runs produced byte-identical CSVs and passed official format/period checks without reference scoring. Anonymous release download matched the frozen archive checksum. Cleanup preserves the frozen runtime and model identities and is checked by an external replay through a clean extraction.

These checks establish reproducibility, not hidden-match accuracy or qualification. See the root AOM.md for provenance and limitations.
