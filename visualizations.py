
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

def main():
    batting_avg = pd.read_csv("output/batting_player_ranking.csv")
    bowling_avg = pd.read_csv("output/bowling_player_ranking.csv")

    plt.title('Actual Batting Avg versus Improved Predicted Batting Avg')
    plt.xlabel('Player Rank')
    plt.ylabel('Batting Average')
    plt.plot(batting_avg['rank'], batting_avg['batting_avg'], 'b.', label="Actual Batting Avg")
    plt.plot(batting_avg['rank'], batting_avg['predicted_bat_avg'], 'r-', label="Predicted Batting Avg")
    legend = plt.legend(loc='upper right')
    
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('white')
    plt.savefig('output/batting.png', dpi=300, bbox_inches='tight')


    plt.title('Actual Bowling Avg versus Improved Predicted Bowling Avg')
    plt.xlabel('Player Rank')
    plt.ylabel('Bowling Average')
    plt.plot(bowling_avg['rank'], bowling_avg['bowling_avg'], 'g.', label="Actual Bowling Avg")
    plt.plot(bowling_avg['rank'], bowling_avg['predicted_bowl_avg'], 'r-', label="Predicted Bowling Avg")
    legend = plt.legend(loc='upper right')
    
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('white')
    plt.savefig('output/bowling.png', dpi=300, bbox_inches='tight')
    

if __name__=='__main__':
    main()