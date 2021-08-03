from typing import Any, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from time import perf_counter
import pickle

#Load data
data = pd.read_csv('Movies_training.csv')
data['Rotten Tomatoes'] = pd.to_numeric(data['Rotten Tomatoes'].str.strip('%')).div(100)
# any=1 cell , all= total cell
#data.dropna(axis = 0 ,how='any', inplace=True)
data['IMDb']=data['IMDb'].fillna(data['IMDb'].mean())
data['Runtime']=data['Runtime'].fillna(data['Runtime'].mean())
data.drop(['Age' ,'Type'],axis=1, inplace=True)

cols = ['Genres' ,'Directors', 'Language', 'Country']
for col in cols:
    data[col].fillna(method='ffill', inplace=True)

movies_data=data.iloc[:,:]
X = data.iloc[:,1:11] #Features
Y = data['IMDb'] #Label
cols = ('Directors','Country','Language')
#convert string to num
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(X[c].values))
    X[c] = lbl.transform(list(X[c].values))


# treat null values
X['Genres'].fillna(' ', inplace = True)
# separate all genres into one list, considering comma + space as separators
genre = X['Genres'].str.split(',')
# flatten the list
flat_genre = [item for sublist in genre for item in sublist]
# convert to a set to make unique
set_genre = set(flat_genre)
# back to list
unique_genre = list(set_genre)
# create columns by each unique genre
Z= X.reindex(X.columns.tolist() + unique_genre, axis=1, fill_value=0)
# for each value inside column, update the dummy
for index, row in Z.iterrows():
    for val in row.Genres.split(','):
        if val != ' ':
            Z.loc[index, val] = 1

Z.drop('Genres', axis = 1, inplace = True)
Z = Z.groupby(Z.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))


#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Z, Y, test_size = 0.30,shuffle=True)
#Get the correlation between the features
corr = movies_data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['IMDb']>0.2)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = movies_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

#calculat algorithm
cls = linear_model.LinearRegression()
#calclute time of fit
Start = perf_counter()
cls.fit(X_train, y_train)
#save model
file_name = 'linrear_regression.sav'
pickle.dump(cls, open(file_name, 'wb'))
End = perf_counter()
time =End -Start

prediction =cls.predict(X_test)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print ('trining time ' ,time ,'second')

true_IMDb = np.asarray(y_test)[0]
predicted_true_IMDb = prediction[0]
print('True  IMDb is : ' + str(true_IMDb))
print('Predicted IMDb  is : ' + str(predicted_true_IMDb))





