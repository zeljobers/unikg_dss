#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:54:22 2024

@author: zorin
"""

import pandas as pd
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
# from sklearn.neighbors import RadiusNeighborsClassifier



from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

def confusionMatrix(y_true,y_pred,title):
	cm=confusion_matrix(y_pred,y_true)
	plt.figure()
	sb.heatmap(cm, annot=True, fmt='0')
	plt.title(title)
	plt.xlabel('True Value')
	plt.ylabel('Predicted Value')
    

dataframe = pd.read_csv('./Social_Network_Ads.csv')
# storing all dataframe without user ids
data = dataframe.iloc[:,1:]

le = LabelEncoder()
le.fit(data.iloc[:, 0])

# Encoding to numerical binary values for the purposes of dealing with NaiveBayes classificators 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

encoded_genders = le.transform(data.iloc[:, 0])
data[data.columns[0]] = encoded_genders
# print(data)
#######################################
# visualization of correlation
sb.pairplot(data, hue="Purchased")
# visualization of distribution against Age feature and target value
sb.displot(data, 
			hue = "Purchased", 
			palette = "Spectral",
			x = data['Age'],
			kind = 'kde',
			fill = True)
# visualization of distribution against EstimatedSalary feature and target value
sb.displot(data, 
			hue = "Purchased", 
			palette = "Spectral",
			x = data['EstimatedSalary'],
			kind = 'kde',
			fill = True)

########################################
x_train, x_test, y_train, y_test = train_test_split(
		data[data.columns[:-1]],
		data[data.columns[-1]],
		test_size=0.2
 	)

print("====================================")
print('Logistic Regression prediction:')
LR = LogisticRegression()
param_grid = {
    'C': np.logspace(-1, -9, num=10),
 	'penalty': ['l1', 'l2', 'elasticnet'],
 	'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'saga'],
 	'class_weight' : ['balanced', 'None', {0: 0.2, 1:0.8}],
 	'l1_ratio': [0, 0.1, 1],
 	'multi_class': ['auto', 'ovr', 'multinomial']
}
LR_grid = GridSearchCV(estimator = LR,
						param_grid=param_grid, verbose=0, 									cv=10, n_jobs=-1)
LR_grid.fit(x_train,y_train)
prediction=LR_grid.predict(x_test)
print('Best case:')
print(LR_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Logistic regression w/ optimization")


print("------------------------------------")
print(f'Logistic Regression prediction without optimization:')
LR.fit(x_train,y_train)
prediction=LR.predict(x_test)

print()
report = classification_report(list(y_test), prediction)
print(report)
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
print("====================================")
print('Gaussian Naive Bayes prediction:')

GNB=GaussianNB()
param_grid = {
    'var_smoothing': np.logspace(1, -9, num=10)
}
GNB_grid = GridSearchCV(estimator = GNB,
						param_grid=param_grid, verbose=0, 											cv=10, n_jobs=-1)
GNB_grid.fit(x_train,y_train)
prediction=GNB_grid.predict(x_test)
print('Best case:')
print(GNB_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)

cm=confusionMatrix(list(y_test),prediction, "Gaussian Naive Bayes w/ optimization")

print("------------------------------------")
print(f'Gaussian Naive Bayes prediction without optimization:')
GNB.fit(x_train,y_train)
prediction=GNB.predict(x_test)
report = classification_report(list(y_test), prediction)
print(report)
print("====================================")

print(f'Multinomial Naive Bayes prediction:')
MNB=MultinomialNB()
param_grid = {
    'alpha': np.logspace(-1, -9, num=10)
}
MNB_grid = GridSearchCV(estimator = MNB,
						param_grid=param_grid, verbose=0, 									 		cv=10, n_jobs=-1)
MNB_grid.fit(x_train,y_train)
prediction=MNB_grid.predict(x_test)
print('Best case:')
print(MNB_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)
print("------------------------------------")
MNB.fit(x_train,y_train)
prediction=MNB.predict(x_test)
print(f'Multinomial Naive Bayes prediction without optimization:')
report = classification_report(list(y_test), prediction)
print(report)
print("====================================")
print(f'Bernoulli Naive Bayes prediction:')
BNB=BernoulliNB()
param_grid = {
    'alpha': np.logspace(-1, -9, num=10)
}
BNB_grid = GridSearchCV(estimator = BNB,
						param_grid=param_grid, verbose=0, 											cv=10, n_jobs=-1)
BNB_grid.fit(x_train,y_train)
prediction=BNB_grid.predict(x_test)
print('Best case:')
print(BNB_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)

print("------------------------------------")
print(f'Bernoulli Naive Bayes prediction without optimization:')
BNB=BernoulliNB()
BNB.fit(x_train,y_train)
prediction=BNB.predict(x_test)

print()
report = classification_report(list(y_test), prediction)
print(report)

print("====================================")

print(f'Categorical Naive Bayes prediction:')

CatNB=CategoricalNB()
param_grid = {
    'alpha': np.logspace(-1, -9, num=10)
}
CatNB_grid = GridSearchCV(estimator = CatNB,
						param_grid=param_grid, verbose=0, 									cv=10, n_jobs=-1)
CatNB_grid.fit(x_train,y_train)
prediction=CatNB_grid.predict(x_test)
print('Best case:')
print(CatNB_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)


CatNB.fit(x_train,y_train)
prediction=CatNB.predict(x_test)
print("------------------------------------")
print(f'Categorical Naive Bayes prediction without optimization:')
CNB=ComplementNB()
CatNB.fit(x_train,y_train)
prediction=CatNB.predict(x_test)

print()
report = classification_report(list(y_test), prediction)
print(report)
print("====================================")
print(f'Complement Naive Bayes prediction:')
CNB=ComplementNB()
param_grid = {
    'alpha': np.logspace(-1, -9, num=10)
}
CNB_grid = GridSearchCV(estimator = CNB,
						param_grid=param_grid, verbose=0, 									cv=10, n_jobs=-1)
CNB_grid.fit(x_train,y_train)
prediction=CNB_grid.predict(x_test)
print('Best case:')
print(CNB_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)


print("------------------------------------")
print(f'Complement Naive Bayes prediction without optimization:')
CNB.fit(x_train,y_train)
prediction=CNB.predict(x_test)

print()
report = classification_report(list(y_test), prediction)
print(report)
print("====================================")
print(f'SVM prediction:')

SVM=SVC()
param_grid = {
    'C': [0.1, 10],
    'gamma': [1, 0.01],
    'kernel': ['rbf', 'linear', 'sigmoid']
}
SVM_grid = GridSearchCV(estimator = SVM,
						param_grid=param_grid, verbose=1, 									cv=3, n_jobs=-1)
SVM_grid.fit(x_train,y_train)
prediction=SVM_grid.predict(x_test)
print('Best case:')
print(SVM_grid.best_estimator_)
print()
report = classification_report(list(y_test), prediction)
print(report)
cm=confusionMatrix(list(y_test),prediction, "SVM w/ optimization")

print("------------------------------------")
print(f'SVM prediction without optimization:')
SVM.fit(x_train,y_train)
prediction=SVM.predict(x_test)

print()
report = classification_report(list(y_test), prediction)
print(report)
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