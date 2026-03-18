import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

"""
Method that returns a dataframe of each March Madness Tournament team from 2013-2025 and how they perform in certain team statistics.
"""
def historical_team_features():
    # BartTorvik Dataset
    barttorvik = pd.read_csv("Dataset/Barttorvik Away-Neutral.csv").sort_values(by = ['SEED', 'YEAR'], ascending = [True, False])
    barttorvik_features = ['TEAM','YEAR', 'TEAM NO', 'SEED', 'BADJ O', 'BADJ D', 'EFG%','EFG%D', 'FTR', 'FTRD', 'FT%', 'TOV%', 'TOV%D', 'OREB%', 'DREB%', 
    '2PT%', '2PT%D', '3PT%', '3PT%D', 'BLK%', 'BLKED%', 'AST%', 'OP AST%', '2PTR', '3PTR', '2PTRD', '3PTRD', 'BADJ T', 'EFF HGT', 'EXP', 'TALENT', 'PPPO', 'PPPD', 'ELITE SOS', 'WAB']
    barttorvik = barttorvik[barttorvik_features]
    
    # EvanMiya Dataset
    evan_miya = pd.read_csv("Dataset/EvanMiya.csv").sort_values(by = ['SEED', 'YEAR'], ascending = [True, False])
    evan_miya_features = ['TEAM', 'YEAR', 'TEAM NO', 'O RATE', 'D RATE','OPPONENT ADJUST', 'PACE ADJUST', 'TRUE TEMPO']
    evan_miya = evan_miya[evan_miya_features]
    
    # KenPom Dataset
    kenpom = pd.read_csv("Dataset/KenPom Barttorvik.csv").sort_values(by = ['SEED', 'YEAR'], ascending = [True, False])
    # Note: O, D, T very similar to barrtorvik - essentially same metric different derived using slightly different methods
    kenpom_features = ['TEAM', 'YEAR', 'TEAM NO', 'KADJ O', 'KADJ D', 'KADJ T']  
    kenpom = kenpom[kenpom_features]
    
    # RPPF Dataset
    rppf = pd.read_csv("Dataset/RPPF Ratings.csv").sort_values(by = ['SEED', 'YEAR'], ascending = [True, False])
    rppf_features = ['TEAM', 'YEAR', 'TEAM NO', 'RADJ O', 'RADJ D', 'R PACE', 'R SOS', 'STROE', 'STRDE']
    rppf = rppf[rppf_features]
    
    # Shooting Splits Dataset
    shooting_splits = pd.read_csv("Dataset/Shooting Splits.csv").sort_values(by = ['YEAR'], ascending = [False])
    shooting_splits_features = ['TEAM', 'YEAR', 'TEAM NO', 'CLOSE TWOS FG%', 'CLOSE TWOS SHARE', 'CLOSE TWOS FG%D', 
    'CLOSE TWOS D SHARE', 'FARTHER TWOS FG%', 'FARTHER TWOS SHARE', 'FARTHER TWOS FG%D', 'FARTHER TWOS D SHARE', 'THREES FG%',
    'THREES SHARE', 'THREES FG%D', 'THREES D SHARE']
    shooting_splits = shooting_splits[shooting_splits_features]
    
    # Creating one df based on our merged team features
    df = barttorvik.merge(evan_miya, on = ['TEAM NO', 'TEAM', 'YEAR'], how = 'inner')
    df = df.merge(kenpom, on = ['TEAM NO', 'TEAM', 'YEAR'], how = 'inner')
    df = df.merge(rppf, on = ['TEAM NO', 'TEAM', 'YEAR'], how = 'inner')
    df = df.merge(shooting_splits, on = ['TEAM NO', 'TEAM', 'YEAR'], how = 'inner')
    mm_team_features = df.groupby(['TEAM', 'YEAR']).mean()

    return mm_team_features

"""
Method that returns a dataframe of all March Madness Tournament matchups from 2013-2025 and the corresponding result.
"""
def historical_matchups():

    # dataset of prior March Madness matchups
    doubled_matchups = pd.read_csv("Dataset/Tournament Matchups.csv")

    # updated to reflect the train/test dataset years
    doubled_matchups = doubled_matchups[doubled_matchups.YEAR >= 2013] 
    
    # only include the columns relevant to a binary (1/0) victory prediction
    doubled_matchups_features = ['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'SCORE']
    doubled_matchups = doubled_matchups[doubled_matchups_features]

    # Create a Game ID we can merge matchups on
    doubled_matchups['GAME ID'] = doubled_matchups.index//2

    # Series of transformations that change 2 adjacent rows/matchup --> 1 row/matchup
    matchups = doubled_matchups.merge(doubled_matchups, how = 'inner', on = 'GAME ID')
    matchups = matchups[matchups['TEAM_x'] != matchups['TEAM_y']]
    matchups = matchups.reset_index()
    matchups = matchups.drop(columns = ['index'])
    matchups = matchups[matchups.index % 2 == 0]
    matchups = matchups.reset_index()
    matchups = matchups.drop(columns = ['index'])

    # Data Cleaning: Having our data be in order --> Better Seeded Team + NO,  Worse Seeded Team + NO
    matchups['Better_Seeded_Team'] = np.where(
        matchups['SEED_x'] <= matchups['SEED_y'], 
        matchups['TEAM_x'],
        matchups['TEAM_y']
    )
    
    matchups['BST_NO'] = np.where(
        matchups['SEED_x'] <= matchups['SEED_y'],
        matchups['TEAM NO_x'],
        matchups['TEAM NO_y']
    )
    
    matchups['Worse_Seeded_Team'] = np.where(
        matchups['SEED_x'] > matchups['SEED_y'],
        matchups['TEAM_x'],
        matchups['TEAM_y']
    )
    
    matchups['WST_NO'] = np.where(
        matchups['SEED_x'] > matchups['SEED_y'],
        matchups['TEAM NO_x'],
        matchups['TEAM NO_y']
    )
    
    # Label that will be used in the train-test process
    matchups['BST'] = ((matchups.SEED_x <= matchups.SEED_y) & (matchups.SCORE_x > matchups.SCORE_y)).astype(int) # if the team with the lower seed (better) has a higher score
    
    # Data Cleaning: removing redundant/useless information:
    matchups = matchups.drop(columns = ['TEAM NO_x', 'TEAM_x', 'SEED_x', 'SCORE_x', 'YEAR_y', 'TEAM NO_y', 'TEAM_y',	'SEED_y', 'SCORE_y']) 
    matchups = matchups.rename(columns = {'YEAR_x': 'YEAR'}) # only one year

    train_WL = matchups['BST_Won'] # train df's future label column
    
    return matchups

"""
Method that returns the dataframe we will be using to train
"""
def difference_features():

    train_diff = []

    for _, team in matchups.iterrows():
        better_seeded_team_number = team['BST_NO']
        worse_seeded_team_number = team['WST_NO']
    
        # Calculate the difference in features
        better_seeded_team_stats = mm_team_features.loc[better_seeded_team_number]
        worse_seeded_team_stats = mm_team_features.loc[worse_seeded_team_number]
    
        # note: no need to flip signs --> model learns on its own
        diff_feature = better_seeded_team_stats - worse_seeded_team_stats # + means advantage for better seeded team; - means advantage for lower seeded team
        train_diff.append(diff_feature.to_list())

        # train df's future feature columns
        train_cols = [x + "_DIFF" for x in mm_team_features.columns.to_list()]
        train = pd.DataFrame(train_diff, columns = train_cols)
        train['BST_Won'] = train_WL

    return train
    
















    
    