# EH t-SNE: Early Hierarchization t-distributed stochastic neighbor embedding

This is a companion repository to our papers:

J.A.Lee et al. Forget early exaggeration in $t$-SNE:\\ early hierarchization preserves global structure
ESANN 2024

J.A.Lee et al. Improving on early exaggeration in $t$-SNE:\\ early hierarchization better preserves global structure
Neurocomputing 2025 (under review)

All codes are in Matlab/Octave m code and C. Some C files should be recompiled with mex or called with their slower m-code counterpart.

# Repository content

The repo contains two folder to be added to your path.

## Folder 'code'

All m-files (and C files).
Function 'basictsne.m' is a quadratic complexity t-SNE (in case Matlab log-linear 'tsne' is not available).
Function 'ehtsne.m' is the one proposed in the two papers.
All other are subfonctions, including 'sbdr_abd_ms.m' (Lee et al. NeuroComputing 2015) and 'nx_scores.m' for quality assessment of dimensionality reduction.
Type 'help <function>' at the prompt for additional information.

## Folder 'data'

A selection of data sets to reproduce the experiments of the second paper with 'EHtsne_runallexp.m' and 'ESANN_art_2024_ext.m'

