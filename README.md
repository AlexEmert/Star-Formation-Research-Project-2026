# Star-Formation-Research-Project-2026
This repo contains all of my work relating to predicting physical properties of yellowballs for my research with Dr. Friedlander and Dr. Devine at the College of Idaho.

----------Important Information---------

`pipeline`: contains the scripts and other files that send modelling jobs to the Borah cluster
  - `/scripts`: contains the scripts that sends jobs to Borah HPC cluster, and fits models using Bayesian Optimization with sk-optimize
  - `/results` and results-related Jupyter notebook files: contains my results and statistical analysis of them

  - `/graphing`: contains a variety of graphs derived from bayesian optimization, and will be updated to include feature importances, response variable distributions, and more
  - `/spaces`: while not the most readable in pkl files (dictionaries found in `scratch/notebooks`), contain the search space of parameters and preprocessing being optimized in search script

`data`: contains all of the cleaned and uncleaned data that will be used, including photometry, distance, cross-matched sources, and anything related to work done on the MIRION catalog

`jobs_and_scripts` (old): first iteration of jobs for borah cluster. inefficient and clunky, which is why it was replaced with pipeline-style scripting

`scratch`: contains various exploratory files, including data cleaning, exploratory data anlysis, and some PCA experimentation

.vscode and catboost_info folders can be ignored
