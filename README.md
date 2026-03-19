# March Madness 2025-26 Model 

Initial commit for a machine learning pipeline to predict NCAA March Madness game outcomes for the 2025-26 Season Tournament.

## Goal
Use historical team-based statistics to build a model that predicts the outcome of all possible games.

## Current Progress
- Collecting and cleaning datasets
- Selecting relevant team statistics
- Merging datasets using a team-season identifier
- Preparing features for matchup-based modeling
- Construct matchup dataset (team A vs team B) --> create difference vector to train on
- Trained models (LogisticRegression, Random Forest, XGBoost) to further enhance feature selection for final model 
  
## Data Sources
Datasets currently used include:
- BartTorvik, EvanMiya, KenPom advanced team statistics
- Shooting split statistics
- Additional efficiency metrics + ranking systems
- used some AI for statistical modeling background info

## Next Steps
- Evaluate model performance after the tournament begins using number of games correctly predicted & Brier Score
- Format/Refactor
- Update directory with final bracket (tree) 