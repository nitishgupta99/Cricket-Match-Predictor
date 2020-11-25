# IPL 2018 Predictor


## Code style
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)
 
## Pre-requisites 
1. Installation of Python (Version 3.7) 
2. Installation of LXML package

## Dependencies
* `NumPy` : For fast matrix operations.
* `pandas` : For analysing and getting insights from datasets.
* `matplotlib` : For creating graphs and plots.
* `seaborn` : For enhancing the style of matplotlib plots.
* `sklearn` : For using the machine learning features
* `re` : For using regex  
* `warnings` : For hiding user warnings
 
## Interface 

### Cricket Wizard 

We created an UI for a good experience for the user.Pictures below show some instances of the main.py while being used. The main.py ,connecting all the branches together, should be run by utilizing python 3.

Menu options after running main.py:

```Python
Welcome to Cricket Wizard! our CMPT 353 Final Projec.

This is an analysis of a method to predict T20 fixture results and its batting/bowling averages:

1- 2018 IPL Match and championship prediction.
2- 2018 IPL Batting/Bowling averages prediction.

Please enter the results that you are intrested to see.(1 or 2):

```
Users can choose 1 for seeing the IPL league predictions and 2 for seeing batting/bowling averages. If the information entered is incorrect users will be asked to  re-enter their response.

```Python
Please enter the results that you are intrested to see.(1 or 2):
D
Wrong entry type! Please re-enter your answer in form of "1" or "2":
```

After correct entry the desired results will be demonstrated and as shown below the user will be asked to choose if they want to run the other branch depending on the branch that they are already running.

```Python
Would you like to see batting averages as well?(y or n):
Would you like to see IPL prediction as well?(y or n):
```

Again if the response in invalid user will be asked to re-enter and this will be the user's final response and they will see the thank you note and the team members names.

```Python
Thank you for using The Cricket Wizard!
```	
	
### 1. IPL Match Predictor

### Parameters

1. Team A
2. Team B
3. Winner

### Return 

A string containing the winner.

### For example 

```
Rajasthan Royals and Mumbai Indians
Winner: Mumbai Indians
```

 
### 2. Bowling/Bating Average Predictor

### Parameters

Input directory containing matches.csv and deliveries.csv. 

### Return 

An output directory containing 6 .csv files containing batting and bowling statistics including: 
* Averages
* Strike Rates
* Improved Predicted Averages 

## Tech/framework used
<b>Built with</b>
- [Python](https://www.python.org/)

## Credits
**Lakshay Sethi , Josie Buter , Pouya Jamali**
