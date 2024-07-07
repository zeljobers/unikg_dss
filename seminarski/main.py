
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

df = pd.read_csv('./data-total.csv')
print(f"Broj nedostajućih vrednosti po kolonama: \n{df.isnull().sum()}")
print()
print("broj duplikata: " + str(len(df[df.duplicated()])))
print()
df = df.drop_duplicates()

df[df.columns[0]] = pd.to_datetime(df.iloc[:,0], format='%m-%d-%Y')
df.sort_values(by=['date', 'time'], inplace=True)

le = LabelEncoder()
le.fit(df.iloc[:, 0])
lookup_dates = df.iloc[:, 0].copy()
encoded_dates = le.transform(df.iloc[:, 0])
df[df.columns[0]] = encoded_dates

ms = MinMaxScaler()
date_scaled = ms.fit_transform(df[[df.columns[0]]])
df[df.columns[0]] = date_scaled.reshape(-1,1)
df[df.columns[1]] = df.iloc[:, 1].map(lambda x: int(x.strip().split(':')[0]))
df[df.columns[1]] = df.iloc[:, 1].map(lambda x: 
									 'breakfast' if 8 <= x < 12 else
									 'lunch' if 12 <= x < 18 else			 
									 'dinner' if 18 <= x < 22 else
									 'bedtime' if 22 <= x <= 23 or 00 <= x < 8 else '')

le.fit(df.iloc[:, 1])
lookup_times = df.iloc[:, 1].copy()

encoded_times = le.transform(df.iloc[:, 1])
df[df.columns[1]] = encoded_times

# print(df)

print(f"Enkodiranje obavljeno nad vremenima:\n {list(set([x for x in zip(lookup_times,encoded_times)]))}")

df[df.columns[-1]] = df.iloc[:, -1].map(lambda x: x if x in ['0Hi', ' 0Hi', ' 0\'\'', ' 0Lo'] else str(float(x)))

print(f" Broj elemenata u skupu podataka: {len(df)}")

# print(len(
#  	[x for x in df[df.columns[-1]] if x not in ['0Hi', ' 0Hi', ' 0\'\'', ' 0Lo']]
#  	))

le.fit(df.iloc[:, -1])
lookup_values = df.iloc[:, 0].copy()
encoded_values = le.transform(df.iloc[:, -1])
df[df.columns[-1]] = encoded_values


print(df)

g = sb.pairplot(df, 
				palette='cubehelix',
				hue='code',
				corner=True
				)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def confusionMatrix(y_true,y_pred,title):
	cm=confusion_matrix(y_pred,y_true)
	plt.figure()
	sb.heatmap(cm, annot=True, fmt='0')
	plt.title(title)
	plt.xlabel('True Value')
	plt.ylabel('Predicted Value')


x_train, x_test, y_train, y_test = train_test_split(
		df[['date', 'time', 'value']],
		df[df.columns[-2]],
		test_size=0.2
 	)


print("====================================")
print('Decision Tree prediction:')
DT = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(3, 15, step=6),
    'min_samples_leaf': np.arange(1, 10, step=3),
	'ccp_alpha': [0, 0.1, 0.2]
}
DT_grid = GridSearchCV(estimator = DT,
					   param_grid=param_grid, verbose=0, 									   cv=10, n_jobs=-1)
DT_grid.fit(x_train,y_train)
prediction=DT_grid.predict(x_test)
print('Best case:')
print(DT_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Decision Tree w/ optimization")


print("------------------------------------")
print(f'Decision Tree prediction without optimization:')
DT.fit(x_train,y_train)
prediction=DT.predict(x_test)

print()
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Random Forst ")

print("====================================")
print('Random Forest prediction:')
RF = RandomForestClassifier()
param_grid = {
    'n_estimators': np.arange(start=4, stop=20, step=4),
    'max_depth': list(range(10, 110, 50)) + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
RF_grid = GridSearchCV(estimator = RF,
						param_grid=param_grid, verbose=0, 											cv=10, n_jobs=-1)
RF_grid.fit(x_train,y_train)
prediction=RF_grid.predict(x_test)
print('Best case:')
print(RF_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Random Forst w/ optimization")

print("------------------------------------")
print(f'Random Forest prediction without optimization:')
RF.fit(x_train,y_train)
prediction=RF.predict(x_test)
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Random Forst ")

print("====================================")
print(f'KNN prediction:')
estimator_KNN = KNeighborsClassifier(algorithm='auto')
parameters_KNN = {
    'n_neighbors': (1,10),
    'leaf_size': (20,40),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev')	   
} 
# with GridSearch
grid_search_KNN = GridSearchCV(
    estimator=estimator_KNN,
    param_grid=parameters_KNN,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
grid_search_KNN.fit(x_train,y_train)
prediction=grid_search_KNN.predict(x_test)
print('Best case:')
print(grid_search_KNN.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)
cm=confusionMatrix(list(y_test),prediction, "KNN w/ optimization")

print("------------------------------------")
print(f'KNN prediction without optimization:')
estimator_KNN = KNeighborsClassifier(algorithm='auto')
estimator_KNN.fit(x_train,y_train)
prediction=estimator_KNN.predict(x_test)
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Random Forst ")