
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)
pd.set_option('mode.chained_assignment', None)

def train_batting_model():
    bat_avg_file = "output/batting_averages.csv"
    bat_avg = pd.read_csv(bat_avg_file)
    bat_strike_file = "output/batting_strike_rate.csv"
    bat_strike = pd.read_csv(bat_strike_file)

    batting = pd.concat([bat_avg, bat_strike], axis=1, join='inner')
    batting = batting.iloc[:, [0, 1, 2, 3, 6, 7]]

    X = batting.iloc[:, [1, 2, 4, 5]].to_numpy()
    y = batting.iloc[:, 3].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = LinearRegression().fit(X_train, y_train)
    # print(reg.score(X_train, y_train), reg.score(X_test, y_test))
    # score is: 0.6636614073163727 0.536466419061135

    poly_reg = make_pipeline(PolynomialFeatures(3), LinearRegression())
    poly_reg.fit(X_train, y_train)
    # print(poly_reg.score(X_train, y_train), poly_reg.score(X_test, y_test))
    # score is: 0.9567347115912241 0.8210366419205732

    return poly_reg

def train_bowling_model():
    bowl_avg_file = "output/bowling_averages.csv"
    bowl_avg = pd.read_csv(bowl_avg_file)
    bowl_strike_file = "output/bowling_strike_rate.csv"
    bowl_strike = pd.read_csv(bowl_strike_file)

    bowling = pd.concat([bowl_avg, bowl_strike], axis=1, join='inner')
    bowling = bowling.iloc[:, [0, 1, 2, 3, 5, 7]]

    X = bowling.iloc[:, [1, 2, 4, 5]].to_numpy()
    y = bowling.iloc[:, 3].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = LinearRegression().fit(X_train, y_train)
    # print(reg.score(X_train, y_train), reg.score(X_test, y_test))
    # score is: 0.938561403144434 0.9612341999504406

    poly_reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_reg.fit(X_train, y_train)
    # print(poly_reg.score(X_train, y_train), poly_reg.score(X_test, y_test))
    # score is: 0.9679470110434435 0.558182264509151

    return reg

def clean_data_to_predict_batting(df):
    # sum runs for each batter
    runs = df.groupby('batsman')['total_runs'].sum()

    # sum outs for each batter
    dismissals = df.iloc[:, [6, 18]] # taking only batters and players dismissed columns
    dismissals['total_dismiss'] = dismissals['batsman'] == dismissals['player_dismissed']
    dismissals['total_dismiss'] = dismissals['total_dismiss'].replace({True: 1, False: 0})
    dismissals = dismissals.groupby('batsman')['total_dismiss'].sum()

    # calculate batting average
    averages = pd.concat([runs, dismissals], axis=1, join='inner')
    averages = averages.drop(averages[(averages.total_dismiss <= 5)].index) # remove batters with 5 or less dismissals
    averages['batting_avg'] = averages['total_runs'] / averages['total_dismiss']

    # sum balls faced
    balls = df.iloc[:, 4:]
    balls['num_balls_faced'] = 1
    balls = balls.groupby(balls['batsman'])['num_balls_faced'].sum()

    # calculate batting strike rate
    bat_strike = pd.concat([runs, balls], axis=1, join='inner')
    bat_strike['strike_rate'] = (bat_strike['total_runs'] / bat_strike['num_balls_faced']) * 100

    # concatenate all features into single dataframe
    batting = pd.concat([averages, bat_strike], axis=1, join='inner')
    batting = batting.reset_index()
    batting = batting.iloc[:, [0, 1, 2, 3, 5, 6]]
    # print(batting)

    X = batting.iloc[:, [1, 2, 4, 5]].to_numpy()
    y = batting.iloc[:, 3].to_numpy()

    return X, y, batting

def clean_data_to_predict_bowling(df):
    # sums runs that were scored by opposite team for each bowler
    runs_conceded = df.groupby('bowler')['total_runs'].sum()

    # sum wickets taken 
    wickets_taken = df.iloc[:, [8, 18]] # taking only bowlers and players dismissed columns
    wickets_taken['total_wickets'] = pd.notnull(wickets_taken.player_dismissed)
    wickets_taken['total_wickets'] = wickets_taken['total_wickets'].replace({True: 1, False: 0})
    wickets_taken = wickets_taken.groupby('bowler')['total_wickets'].sum()

    # calculate bowling average
    averages = pd.concat([runs_conceded, wickets_taken], axis=1, join='inner')
    averages = averages.drop(averages[(averages.total_wickets <= 0)].index) # remove players with 0 wickets taken
    averages['bowling_avg'] = averages['total_runs'] / averages['total_wickets']

    # sum num balls bowled
    balls_bowled = df.iloc[:, 4:]
    balls_bowled['num_bowled'] = 1
    balls_bowled = balls_bowled.groupby(balls_bowled['bowler'])['num_bowled'].sum()

    # calculate bowling strike rate
    bowl_strike = pd.concat([balls_bowled, wickets_taken], axis=1, join='inner')
    bowl_strike = bowl_strike.drop(bowl_strike[(bowl_strike.total_wickets <= 0)].index) # remove players with 0 wickets taken
    bowl_strike['strike_rate'] = bowl_strike['num_bowled'] / bowl_strike['total_wickets']

    # concatenate all features into single dataframe
    bowling = pd.concat([averages, bowl_strike], axis=1, join='inner')
    bowling = bowling.reset_index()
    bowling = bowling.iloc[:, [0, 1, 2, 3, 4, 6]]
    # print(bowling)

    X = bowling.iloc[:, [1, 2, 4, 5]].to_numpy()
    y = bowling.iloc[:, 3].to_numpy()

    return X, y, bowling

def main():
    batting_ranking_2016_file = "input/2016_batting_ranking.csv"
    batting_rank_2016 = pd.read_csv(batting_ranking_2016_file)
    bowling_ranking_2016_file = "input/2016_bowling_ranking.csv"
    bowling_rank_2016 = pd.read_csv(bowling_ranking_2016_file)

    deliveries_file = "input/deliveries.csv"
    deliveries = pd.read_csv(deliveries_file)

    # keeping only 2016 data to predict batting/bowling averages 
    del_2016 = deliveries.drop(deliveries[(deliveries.match_id < 577) | (deliveries.match_id > 636)].index)

    # get data for model to predict averages and dataframe with player names to map back to
    X_bat, y_bat, batting_stats = clean_data_to_predict_batting(del_2016)
    X_bowl, y_bowl, bowling_stats = clean_data_to_predict_bowling(del_2016)

    # predict averages with trained models
    bat_model = train_batting_model()
    bowl_model = train_bowling_model()
    bat_predict = bat_model.predict(X_bat)
    bowl_predict = bowl_model.predict(X_bowl)

    # add predicted averages to table with player names and original stats and sorting by bat/bowl average
    batting_stats['predicted_bat_avg'] = bat_predict
    batting_stats = batting_stats.sort_values(by=['predicted_bat_avg'], ascending=False)
    batting_stats = batting_stats.reset_index()
    batting_stats = batting_stats.drop(columns=['index'])
    bowling_stats['predicted_bowl_avg'] = bowl_predict
    bowling_stats = bowling_stats.sort_values(by=['predicted_bowl_avg'], ascending=False)
    bowling_stats = bowling_stats.reset_index()
    bowling_stats = bowling_stats.drop(columns=['index'])

    # add ranks to our dataframe
    batting_stats['rank'] = batting_stats['predicted_bat_avg'].rank(ascending=False)
    bowling_stats['rank'] = bowling_stats['predicted_bowl_avg'].rank(ascending=False)

    # re-arrange columns 
    batting_stats = batting_stats[['rank', 'batsman', 'total_runs', 'total_dismiss', 'num_balls_faced', 'strike_rate', 'batting_avg', 'predicted_bat_avg']]
    bowling_stats = bowling_stats[['rank', 'bowler', 'total_runs', 'total_wickets', 'num_bowled', 'strike_rate', 'bowling_avg', 'predicted_bowl_avg']]

    print("_____Batting Stats with Model Prediction_____")
    print(batting_stats)
    # print(batting_rank_2016)
    print("_____Bowling Stats with Model Prediction_____")
    print(bowling_stats)
    # print(bowling_rank_2016)

    batting_stats.to_csv("output/batting_player_ranking.csv")
    bowling_stats.to_csv("output/bowling_player_ranking.csv")

    

if __name__=='__main__':
    main()