import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#base elo score 1600
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
#the folder that keeps data
folder = 'data'

#initialize team stats according to each teams' Miscellaneous Opponent 
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')
    return team_stats1.set_index('Team', inplace=False, drop=True)

#get elo score for the team
def get_elo(team):
    try:
        return team_elos[team]
    except:
        #if the team has no elo, give it the base elo score of 1600
        team_elos[team] = base_elo
        return team_elos[team]

# calculate the elo score of each team 
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # base on rules of elo rating, change k's value according to rank
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16

    #update the new rank 
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_loser_rank = round(loser_rank + (k * (0 - odds)))
    return new_winner_rank, new_loser_rank

#use data and elo score to build a dataset
def  build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    for index, row in all_data.iterrows():

        Wteam = row['WTeam']
        Lteam = row['LTeam']

        #get the initial elo of each team
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        #add 100 elo score to the home team
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        #set elo score as the first feature in evaluating each team
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        #append the data gathered from basketball reference.com
        for key, value in team_stats.loc[Wteam].items():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].items():
            team2_features.append(value)

        #spread the feature value randomly on the two teams of the match and give the corresponding 0 or 1 to y
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        if skip == 0:
            print('X',X)
            skip = 1

        #update the elo score accordingly
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), y

if __name__ == '__main__':

    Mstat = pd.read_csv(folder + '/15-16Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/15-16Opponent_Per_Game_Stat.csv')
    Tstat = pd.read_csv(folder + '/15-16Team_Per_Game_Stat.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/2015-2016_result.csv')
    X, y = build_dataSet(result_data)

    #training the model
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    #use 10-fold cross validation
    print("Doing cross-validation..")
    print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())


    def predict_winner(team_1, team_2, model):
        features = []

        # team 1，away team
        features.append(get_elo(team_1))
        for key, value in team_stats.loc[team_1].items():
            features.append(value)

        # team 2，home team
        features.append(get_elo(team_2) + 100)
        for key, value in team_stats.loc[team_2].items():
            features.append(value)

        features = np.nan_to_num(features)
        return model.predict_proba([features])


    #use the trained model to predict the matches for season 16-17

    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(folder + '/16-17Schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])

    #write the prediciton into 16-17Result.csv
    with open('16-17Result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)
        print('done.')

        pd.read_csv('16-17Result.csv', header=0)
