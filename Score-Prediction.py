
import pandas as pd
import numpy as np


data = pd.read_csv('data.csv')
data.head()


# selecting features columns

# features
X = data[['bat_team', 'bowl_team', 'runs', 'wickets', 'overs', "runs_last_5", "wickets_last_5", "total"]]
X.head()


current_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab',
                 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']


# selecting teams that are currently playing in the ipl
X =  X[(X['bat_team'].isin(current_teams)) & (X['bowl_team'].isin(current_teams))]


Y = X[["total"]] # total score
X = X[['bat_team','bowl_team', 'runs','wickets', 'overs', "runs_last_5", "wickets_last_5"]] # features


# converted string team names to integer value
d = {'bat_team' : {'Delhi Daredevils' : 1, 'Royal Challengers Bangalore' : 2, 'Rajasthan Royals' : 3, 'Mumbai Indians' : 4,
                    'Kings XI Punjab' : 5, 'Kolkata Knight Riders' : 6, 'Chennai Super Kings' : 7, 'Sunrisers Hyderabad' : 8},
                   
     'bowl_team' : {'Delhi Daredevils' : 1, 'Royal Challengers Bangalore' : 2, 'Rajasthan Royals' : 3, 'Mumbai Indians' : 4,
                   'Kings XI Punjab' : 5, 'Kolkata Knight Riders' : 6, 'Chennai Super Kings' : 7, 'Sunrisers Hyderabad' : 8},
     }


# replacing columns values
X = X.replace(d)


# updated data
X.head()


Y.head()


X = np.array(X)
Y = np.array(Y)
# splitting the data into training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10)
# 70% for training and 30% for testing


# scales the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# model1
# create and train the model
from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor(n_estimators=100, max_depth= None, random_state = 10)
model1.fit(X_train,y_train.ravel())


# predict score for the test data
y_pred1 = model1.predict(X_test)


# function to calculate accuracy
def accuracy(y_test,y_pred,thresold):
    correct_prediction = 0
    n = len(y_pred)
    
    for i in range(n):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            correct_prediction += 1
    
    return ((correct_prediction/n)*100)


print("Accuracy:" , accuracy(y_test,y_pred1,15))


from sklearn.metrics import r2_score
print("R-squared value :", r2_score(y_test, y_pred1) * 100)


# predicting score value for new features
predict_score = model1.predict(sc.transform(np.array([[5, 1, 121, 4, 16.1, 50, 2]])))
score = int(*predict_score)

print("Predicted First Innings Score is in the range :" , score - 5, "to", score + 5)

























