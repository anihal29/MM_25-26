# March Madness 2025-26 Model (SAC)

Machine learning pipeline to predict NCAA March Madness game outcomes for the 2025-26 Season Tournament.

## Authors
Ayaan Nihal

## Goal
Use historical team-based statistics to build a model that predicts the outcome of all possible games. Primary evaluation done with Brier Score*. <br><br>
<i>The Brier score is calculated as the average of the squared differences between the predicted probabilities and the actual outcomes (where outcomes are 0 or 1).*</i>

## Steps
- Collected and cleaned datasets
- Selected relevant team statistics
- Merged datasets using a team-season identifier
- Prepared features for matchup-based modeling
- Constructed matchup dataset (team A vs team B) --> created difference vector to train on
- Trained models (LogisticRegression, Random Forest, XGBoost) to further enhance feature selection for final model
- Retrained model on all historical data from 2013-25 (RandomForest) and predicted matchups for all potential 2026 March Madness games
- Evaluated model performance after the tournament using the Brier Score (& numer of games correctly predicted)
- Updated directory with final bracket (tree) using both March Madness Tournament Challenge & Per-Game statistics

## Results
- <a href = "Results/README.md">Achieved a Brier Score of 0.1654</a>
  
## Data Sources
Datasets used from [Nishaan Amin](https://www.kaggle.com/datasets/nishaanamin/march-madness-data/data):
- BartTorvik, EvanMiya, KenPom, RPPF advanced team statistics
- Shooting Split statistics
- Official March Madness Bracket
