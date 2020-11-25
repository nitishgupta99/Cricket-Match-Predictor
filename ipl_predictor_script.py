import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn.model_selection import cross_val_score
import warnings


warnings.simplefilter(action='ignore', category=UserWarning)


# Retriving the schedule for the whole tournament.
match_list = pd.read_html("https://www.icccricketschedule.com/vivo-ipl-2018-schedule-fixture/", header=0)
match = match_list[0]

# Retrieving past Ipl Match results.
raw_data = pd.read_csv("input/matches.csv")
data = raw_data[['id' , 'season' , 'team1' , 'team2' , 'winner']]
before_1617 = data.query('season != "2018" and season != "2019"')
before_1617 = before_1617[before_1617['winner'].notnull()]

no_duplicates = before_1617.drop_duplicates(subset = "id") 
no_duplicates = no_duplicates.sort_values('season')


# Removing the now defunct teams from the data set.
worldcup_teams = ['Chennai Super Kings', 'Mumbai Indians', 'Rajasthan Royals',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
       'Delhi Daredevils']
no_duplicates = no_duplicates[no_duplicates['team1'].isin(worldcup_teams)]
no_duplicates = no_duplicates[no_duplicates['team2'].isin(worldcup_teams)]
no_duplicates = no_duplicates.drop(columns = ['id', 'season'])

# Converting the labels into numeric data for modelling.
converted = pd.get_dummies(no_duplicates, prefix=['team1', 'team2'], columns=['team1', 'team2'])

X = converted.drop(columns = ['winner'])
y = converted["winner"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_rgb:.3f} \n'
    'kNN classifier:         {knn_rgb:.3f} \n'
    'Rand forest classifier: {rf_rgb:.3f} \n'
)


bayes_rgb_model = GaussianNB()
    
knn_rgb_model = KNeighborsClassifier(n_neighbors= 15)
    
rf_rgb_model =  RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=10)
    
    
# Various models trained
models = [ bayes_rgb_model, knn_rgb_model, rf_rgb_model ]
for i,m in enumerate(models):  
    m.fit(X_train, y_train)
        
print(OUTPUT_TEMPLATE.format(
    bayes_rgb=bayes_rgb_model.score(X_valid, y_valid),
        
    knn_rgb=knn_rgb_model.score(X_valid, y_valid),
        
    rf_rgb=rf_rgb_model.score(X_valid, y_valid)
))


########################################  Qualifiers ##################################################################
match['Date'] = match['Date'].str.split('-', expand = True)[2]

match['Team1'] = match['IPL 2018 Teams'].str.split(' v | vs | Vs ', expand = True)[0]
match['Team2'] = match['IPL 2018 Teams'].str.split(' v | vs | Vs ', expand = True)[1]

match = match.drop(['No' , 'Time' , 'IPL 2018 Teams' , 'Date' , 'Stadium / Venues'], axis = 1)
match = match[match['Team1'] != 'M.A.Chidambaram Stadium Chennai Tamil Nadu']
match = match.replace(['Delhi Dare Devils' , 'Delhi Dare Devils Wankhede'  ], 'Delhi Daredevils')
match = match.replace('Sun Risers Hyderabad' , 'Sunrisers Hyderabad')
match = match.replace('Chennai Supers Kings' , 'Chennai Super Kings')
match = match.replace('RCB' , 'Royal Challengers Bangalore')

# Converting categorial data into numeric for Prediction
pred_ready_match = pd.get_dummies(match, columns=["Team1", "Team2"], prefix=["team1", "team2"])

# Score Table
score_table = pd.DataFrame(worldcup_teams ,columns= [ 'Team'])
score_table['Score'] = np.repeat(0 , 8)
table = score_table.pivot(index='Score', columns='Team', values='Score')
table.index.name = None

# Predicting results of the qualifiers.
predictions = rf_rgb_model.predict(pred_ready_match)

print('########################################  Qualifiers ##################################################################')

for i in range(match.shape[0]):
    print(match.iloc[i, 1] + " and " + match.iloc[i, 0])
    if predictions[i] == 1:
        table[match.iloc[i, 1]] = table[match.iloc[i, 1]] + 1
        print("Winner: " + match.iloc[i, 1])
    
    else:
        table[match.iloc[i, 0]] = table[match.iloc[i, 0]] + 1
        print("Winner: " + match.iloc[i, 0])
    print("")

qualifiers = table.iloc[0 , :].sort_values( ascending=False)
qualifiers.index.name = None
semifinals = pd.DataFrame({ 'Position' : qualifiers[0 : 4] }  )
semifinals['Position'] = semifinals['Position'].apply(str)

# Teams which qualified to the semi finals
teams = semifinals.index.values

########################################  Semi Finals ##################################################################
semiFinal = pd.DataFrame({'team1' : [teams[0] ,teams[2]]})
semiFinal['team2'] = [teams[1] ,teams[3]]

# Converting categorial data into numeric for Prediction
pred_semi_match = pd.get_dummies(semiFinal, columns=["team1","team2"], prefix=['team1' , 'team2'])

# Add missing columns compared to the model's training dataset
missing_cols2 = set(pred_ready_match.columns) - set(pred_semi_match.columns)
for d in missing_cols2:
    pred_semi_match[d] = 0
pred_semi_match = pred_semi_match[pred_ready_match.columns]
semi_result = rf_rgb_model.predict(pred_semi_match)

print('########################################  Semi Finals ##################################################################')

for i in range(2):
    print(semiFinal['team1'][i] + " and " + semiFinal['team2'][i])
    if semi_result[i] == semiFinal['team1'][i] :
        print("Winner: " +semi_result[i])
    
    else:
        print("Winner: " + semi_result[i])
    print("")
    
########################################  Finals ##################################################################
final = pd.DataFrame({'team1' : [semi_result[0]]})
final['team2'] = [semi_result[1]]

# Converting categorial data into numeric for Prediction
pred_final_match = pd.get_dummies(final, columns=["team1","team2"], prefix=['team1' , 'team2'])

# Add missing columns compared to the model's training dataset
missing_cols3 = set(pred_ready_match.columns) - set(pred_final_match.columns)
for d in missing_cols3:
    pred_final_match[d] = 0
pred_final_match = pred_final_match[pred_ready_match.columns]
final_result =rf_rgb_model.predict(pred_final_match)

print('########################################  Finals ##################################################################')

print(semi_result[0] + " and " + semi_result[1])
print("Winner: " +final_result[0])
print("")