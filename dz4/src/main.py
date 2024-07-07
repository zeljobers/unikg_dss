import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from sklearn.metrics import r2_score
import warnings 
warnings.filterwarnings("ignore")


def evaluate(y_pred, y_test,name):
    
    square_error=np.array(y_test-y_pred)
    square_error=np.power(square_error,2)
    square_error=np.sum(square_error)
    
    mean_square_error=square_error/len(y_pred)
    
    root_mean_square_error=np.sqrt(mean_square_error)
    
    print(name)
    print('mean square error: ' + str(round(mean_square_error,2)))
    print('root mean square error: ' + str(round(root_mean_square_error,2)))
    print()
    
    return mean_square_error,root_mean_square_error

dataframe = pd.read_csv('./housing.csv')


print("==================")
print("Nedostajuće vrednosti:")
print(dataframe.isnull().sum())
df1 = dataframe[dataframe['total_bedrooms'].notnull()]

print("==================")
duplikati = df1[df1.duplicated()]
if len(duplikati) == 0:
	print("Nema duplikata")
else:
	print(duplikati)
print("==================")	

df1['ocean_proximity']= LabelEncoder().fit_transform(df1['ocean_proximity']) 

sns.pairplot(df1, palette="Spectral", hue="median_house_value", dropna=True)
plt.show()

y1 = df1['median_house_value']
x1 = df1[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]]
sc = MinMaxScaler()
x1_scalled = sc.fit_transform(x1)


X_train, X_test, y_train, y_test = train_test_split(x1_scalled, y1, test_size = 0.2)
########################################
model = LinearRegression()

model.fit(X_train[:,0].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,0].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over longitude")
########################################
model.fit(X_train[:,1].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,1].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over latitude")
########################################
model.fit(X_train[:,2].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,2].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over housing_median_age")
########################################
model.fit(X_train[:,3].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,3].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over total_rooms")
########################################
model.fit(X_train[:,4].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,4].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over total_bedrooms")
########################################
model.fit(X_train[:,5].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,5].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over population")
########################################
model.fit(X_train[:,6].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,6].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over households")
########################################
model.fit(X_train[:,7].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,7].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over median_income")

print(f"R2 score: {r2_score(y_test, predicted)}")
plt.figure()
plt.title('regression visualisation')
plt.scatter(X_test[:,7], y_test, color='black')
plt.plot(X_test[:,7], predicted)
plt.show()

plt.figure()
plt.title('linear model visualisation')
plt.scatter(X_test[:,7], y_test, color='black')
coefficients = model.coef_
intercept = model.intercept_
k = coefficients
n = intercept

line_x = np.arange(min(X_train[:,7]), 1.01*max(X_train[:,7]), (max(X_train[:,7])- min(X_train[:,7]))/10)
line_y = k * line_x + n
plt.plot(line_x, line_y, color='red')
plt.show()

########################################
model.fit(X_train[:,8].reshape(-1,1),y_train)
predicted = model.predict(X_test[:,8].reshape(-1,1))

mse,rmse=evaluate(predicted, list(y_test), "Simple linear regresion over ocean_proximity")
########################################
print("==================")
model = LinearRegression()
model.fit(X_train,y_train)
predicted = model.predict(X_test)

mse,rmse=evaluate(predicted, list(y_test), "Multiple linear regresion")
print(f"R2 score: {r2_score(y_test, predicted)}")


plt.figure()
plt.title('linear model visualisation')
plt.scatter(X_test, y_test, color='black')
coefficients = model.coef_
intercept = model.intercept_
k = coefficients
n = intercept

line_x = np.arange(min(X_train), 1.01*max(X_train), (max(X_train)- min(X_train))/10)
line_y = k * line_x + n
plt.plot(line_x, line_y, color='red')
plt.show()
print("==================")
# https://www.kaggle.com/code/ismailaitalimhamed/california-housing-prices-linear-polynomial-model
polynomialRegression = PolynomialFeatures(degree = 3)
X_train_poly = polynomialRegression.fit_transform(X_train)
X_test_poly = polynomialRegression.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)

predicted = model.predict(X_test_poly)

mse, rmse = evaluate(list(y_test), predicted, 'polynomial_regression')
print(f"R2 score: {r2_score(y_test, predicted)}")

# NOTE Postoji SGDRegression #########################################


############################
plt.figure()
plt.title('polynomial regr visualization')
plt.scatter (X_test[:,6], y_test, color="black")

coefficients = model.coef_
intercept = model.intercept_
k1 = coefficients[0]
k2 = coefficients[1]
k3 = coefficients[2]
n = intercept

X = X_train[:,0]
line_x1 = np.arange(min(X), 
 					max(X) * 1.01, 
 					(max(X) - min(X)) / 10)
line_x2 = line_x1 ** 2
line_x3 = line_x1 ** 3

line_y3 = k1*line_x1 + k2*line_x2 + k3*line_x3 + n

plt.plot(line_x1, line_y3, color="red")
plt.show()


##############################
decisionTreeRegressor=DecisionTreeRegressor()
decisionTreeRegressor.fit(X_train, y_train)
predicted = decisionTreeRegressor.predict(X_test)
mse, rmse = evaluate(list(y_test), predicted, 'decision tree')
print(f"R2 score: {r2_score(y_test, predicted)}")

print("------------------")
# https://www.nbshare.io/notebook/312837011/Decision-Tree-Regression-With-Hyper-Parameter-Tuning-In-Python/
params={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            "max_features":["auto","log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]}

modelGrid=GridSearchCV(DecisionTreeRegressor(),param_grid=params,cv=5)
modelGrid.fit(X_train,y_train)
modelBest = modelGrid.best_estimator_
predicted=modelBest.predict(X_test)
mse,rmse=evaluate(predicted, list(y_test), "decision tree with hyperparameter optimization")
print(f"R2 score: {r2_score(y_test, predicted)}")
print('best params after grid search: ')
print(modelGrid.best_params_)


############################
print("==================")
randomForestRegressor=RandomForestRegressor(n_estimators=10,random_state=0)
randomForestRegressor.fit(X_train, y_train)
predicted = randomForestRegressor.predict(X_test)
mse, rmse = evaluate(list(y_test), predicted, 'random forest')
print(f"R2 score: {r2_score(y_test, predicted)}")
print("------------------")
# https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/
params={'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9]
 	}
            
modelGrid=GridSearchCV(RandomForestRegressor(),param_grid=params,cv=5)
modelGrid.fit(X_train,y_train)
modelBest = modelGrid.best_estimator_
predicted=modelBest.predict(X_test)
mse,rmse=evaluate(predicted, list(y_test), "Random Forest with hyperparameter optimization")
print(f"R2 score: {r2_score(y_test, predicted)}")
print('best params after grid search: ')
print(modelGrid.best_params_)
print("==================")
supportVectorRegressor=SVR(kernel='rbf')
supportVectorRegressor.fit(X_train,y_train)
predicted = supportVectorRegressor.predict(X_test)
mse, rmse = evaluate(list(y_test), predicted, 'svr')
print(f"R2 score: {r2_score(y_test, predicted)}")

print("------------------")
# https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/
params = {'C': [0.1, 10, 100],  
              'gamma': [1, 0.01, 0.001], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  


modelGrid=GridSearchCV(SVR(),param_grid=params,cv=5)
modelGrid.fit(X_train,y_train)
modelBest = modelGrid.best_estimator_
predicted=modelBest.predict(X_test)
mse,rmse=evaluate(predicted, list(y_test), "Support vector regression with hyperparameter optimization")
print(f"R2 score: {r2_score(y_test, predicted)}")
print('best params after grid search: ')
print(modelGrid.best_params_)
print("==================")
