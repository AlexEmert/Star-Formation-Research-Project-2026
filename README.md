# Star-Formation-Research-Project-2026
This repo contains all of my work relating to predicting physical properties of yellowballs for my research with Dr. Friedlander and Dr. Devine at the College of Idaho during the summer of 2026.

----------Important Information---------

`models`: contains the majority of the relevant scripts and modeling information, split into 3 folders.
  - `full_models`: contains the scripts that created the optimized models containing ALL available wavlengths, up to 1100 microns. Note, these include cross matched sources. The results from these models is contained in `results_final_6-25-26.csv` in the above directory. These models were created using BayesSearchCV in skopt.
  - `predictive_models`: contains the scripts that created the optimized models for ONLY the 4 MIRION fluxes, which were used to predict the properties for the rest of the yellowballs in the catalog. These predictions are stored in the folder `predictions`. These models were created near the end of summer after learning the syntax for Optuna, a more flexible optimization package. 
  - `old_models`: contains a variety of not very organized intermediate models that are not the most optimized given the overall information. Within the old models directory, there are the following subdirectories:
      - `5-25-26_intial_optimized_models`: the success of these models occured due to an error in which the response variable was included in the model as a feature.
      - `first_optuna_models`: includes all of the scripts used to learn the syntax for optuna. A variety of options and pipelines were tested, with the best one used for the predictive models. Options tested included an expanding boundary function that did not end up working in the `ratio_function.py` file, and more.
      - other subdirectories consist of inefficient models and other scratch testing that was reworked before creating the final full and predictive models.
   
`graphing`: contains a variety of graphing scripts

`data`: contains both clean and unclean data, along with the main script used to the clean said data. The most used csv that contains all the relevant information is `MIRION_cleaned_everything.csv`.

`predictions`: contains the predicted intervals for the remainder of the yellowballs in the MIRION catalog unable to be crossmatched. Each csv corresponds to a different physical property. 

`scratch`: contains a variety of testing and other work, incuding PCA, SVR, initial Catboost models, and more. 
