
import numpy as np
import pandas as pd  
pd.set_option('mode.chained_assignment', None)

'''
Bowling average is calculated for the period of 2008-2015, so all the runs and dismissals are summed, then
the division of runs conceded/wickets taken is calculated.

Bowling Average = Runs conceded / Wickets taken*

*Wickets taken is the same thing as players dismissed while player is bowling.
'''

def main():
    # reading in the ball by ball dataset (deliveries.csv)
    deliveries_file = "input/deliveries.csv"
    deliveries = pd.read_csv(deliveries_file)

    # keeping only data from 2008 - 2015
    deliveries = deliveries.drop(deliveries[(deliveries.match_id < 60) | (deliveries.match_id > 576)].index)

    # sums runs that were scored by opposite team for each bowler
    runs_conceded = deliveries.groupby('bowler')['total_runs'].sum()

    # sum wickets taken 
    wickets_taken = deliveries.iloc[:, [8, 18]] # taking only bowlers and players dismissed columns
    wickets_taken['total_wickets'] = pd.notnull(wickets_taken.player_dismissed)
    wickets_taken['total_wickets'] = wickets_taken['total_wickets'].replace({True: 1, False: 0})
    wickets_taken = wickets_taken.groupby('bowler')['total_wickets'].sum()

    # calculate bowling average
    averages = pd.concat([runs_conceded, wickets_taken], axis=1, join='inner')
    averages = averages.drop(averages[(averages.total_wickets <= 0)].index) # remove players with 0 wickets taken
    averages['bowling_avg'] = averages['total_runs'] / averages['total_wickets']

    print("_____Bowling Averages_____")
    print(averages)

    averages.to_csv("output/bowling_averages.csv")



if __name__=='__main__':
    main()