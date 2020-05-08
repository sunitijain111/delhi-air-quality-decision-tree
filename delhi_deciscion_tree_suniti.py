import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

#reading dataset
df1= pd.read_excel('delhi air 1996 to 2015.xls')
df2= pd.read_excel('delhi weather since 1997.xlsx') 
df3= pd.merge(df1, df2, how='inner', on=['month','year'])
df3.fillna(df3.mean(), inplace= True)
df3=df3.drop('PM 2.5', axis=1)

#preparing dataset
X = df3.drop(['month','year','temp'], axis=1)
y = df3['temp']

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training algo
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)         

#predicting results
y_pred = regressor.predict(X_test)
print(y_pred)

#visualizing results of one feature(NO2) and label
Z=df3.iloc[:,1:2].values #used for visualizing 
reg= DecisionTreeRegressor()
reg.fit(Z,y)
X_grid = np.arange(min(Z), max(Z), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(Z, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')

#evaluating model
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
 

print("Accuracy in r- square :",regressor.score(X_test, y_test))


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
feature_cols=['NO2','RSPM/PM10','SO2','SPM']
export_graphviz(regressor, out_file=dot_data,   filled=True, rounded=True,special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('delhi climate.png')
Image(graph.create_png())
