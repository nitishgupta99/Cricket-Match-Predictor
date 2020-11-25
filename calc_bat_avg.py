
import numpy as np
import pandas as pd  
pd.set_option('mode.chained_assignment', None)

'''
Batting average is calculated for the period of 2008-2015, so all runs and dismissals from all games are 
summed, then the division of runs/dismissals is calculated. We have removed batters with 5 or less dismissals 
because it tells us they aren't valuable batters for their team since they are not batting early enough in 
the innings to be dismissed. (Taken from the "Handbook of Statistical Methods and Analyses in Sports" in the 
Evaluating Player Performance section of the Cricket chapter.)

Batting Average = Runs scored / Number times dismissed 
'''

def main():
    # reading in the ball by ball dataset (deliveries.csv)
    deliveries_file = "input/deliveries.csv"
    deliveries = pd.read_csv(deliveries_file)

    # keeping only data from 2008 - 2015
    deliveries = deliveries.drop(deliveries[(deliveries.match_id < 60) | (deliveries.match_id > 576)].index)

    # sum runs for each batter
    runs = deliveries.groupby('batsman')['total_runs'].sum()

    # sum outs for each batter
    dismissals = deliveries.iloc[:, [6, 18]] # taking only batters and players dismissed columns
    dismissals['total_dismiss'] = dismissals['batsman'] == dismissals['player_dismissed']
    dismissals['total_dismiss'] = dismissals['total_dismiss'].replace({True: 1, False: 0})
    dismissals = dismissals.groupby('batsman')['total_dismiss'].sum()

    # calculate batting average
    averages = pd.concat([runs, dismissals], axis=1, join='inner')
    averages = averages.drop(averages[(averages.total_dismiss <= 5)].index) # remove batters with 5 or less dismissals
    averages['batting_avg'] = averages['total_runs'] / averages['total_dismiss']

    print("_____Batting Averages_____")
    print(averages)

    averages.to_csv("output/batting_averages.csv")

if __name__=='__main__':
    main()