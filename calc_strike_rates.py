
import numpy as np
import pandas as pd  
pd.set_option('mode.chained_assignment', None)

'''
Batting Strike Rate = (runs scored/balls faced)*100

Bowling Strike Rate = (# balls bowled / # wickets taken)
'''

def main():
    # reading in the ball by ball dataset (deliveries.csv)
    deliveries_file = "input/deliveries.csv"
    deliveries = pd.read_csv(deliveries_file)

    # keeping only data from 2008 - 2015
    deliveries = deliveries.drop(deliveries[(deliveries.match_id < 60) | (deliveries.match_id > 576)].index)

    # Calculate Batting Strike Rate

    # sum runs for each batter
    runs = deliveries.groupby('batsman')['total_runs'].sum()

    # sum balls faced
    balls = deliveries.iloc[:, 4:]
    balls['num_balls_faced'] = 1
    balls = balls.groupby(balls['batsman'])['num_balls_faced'].sum()

    # calculate batting strike rate
    bat_strike = pd.concat([runs, balls], axis=1, join='inner')
    bat_strike['strike_rate'] = (bat_strike['total_runs'] / bat_strike['num_balls_faced']) * 100

    print("_____Batting Strike Rates_____")
    print(bat_strike)

    # Calculate Bowling Strike Rate

    # sum num balls bowled
    balls_bowled = deliveries.iloc[:, 4:]
    balls_bowled['num_bowled'] = 1
    balls_bowled = balls_bowled.groupby(balls_bowled['bowler'])['num_bowled'].sum()

    # sum wickets taken
    wickets_taken = deliveries.iloc[:, [8, 18]] # taking only bowlers and players dismissed columns
    wickets_taken['total_wickets'] = pd.notnull(wickets_taken.player_dismissed)
    wickets_taken['total_wickets'] = wickets_taken['total_wickets'].replace({True: 1, False: 0})
    wickets_taken = wickets_taken.groupby('bowler')['total_wickets'].sum()

    # calculate bowling strike rate
    bowl_strike = pd.concat([balls_bowled, wickets_taken], axis=1, join='inner')
    bowl_strike = bowl_strike.drop(bowl_strike[(bowl_strike.total_wickets <= 0)].index) # remove players with 0 wickets taken
    bowl_strike['strike_rate'] = bowl_strike['num_bowled'] / bowl_strike['total_wickets']

    print("_____Bowling Strike Rates_____")
    print(bowl_strike)

    # write dataframes to .csv files
    bat_strike.to_csv("output/batting_strike_rate.csv")
    bowl_strike.to_csv("output/bowling_strike_rate.csv")



if __name__=='__main__':
    main()