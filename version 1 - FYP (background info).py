import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.metrics import accuracy_score
#from __future__ import print_function

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#---Do Update for changes.
directory="C:\\Users\\Administrator\\Desktop\\FYP\\Data"
spreadsheet = 'BigData - For Student.xlsx'
spreadsheet1 = 'Output_all.xlsx'
##'BG_Cat','BG_US_P','BG_US_A','BG_CH','BG_SC_A','BG_SC_P','BG_S','BG_Yr','BG_N'

result = ['E_Total\nWk 17\n(60%)','Overall\nWk 17']
# set directory
os.chdir(directory)

# df <- read excel
df = pd.read_excel(open(spreadsheet,'rb'), sheetname='BigData (Student)')
##xl = pd.read_excel(spreadsheet)

# filter data | remove Nan/Null in column 'BG_Cat' 
df.dropna(subset=['BG_Cat'], inplace=True)
df=df[df['Overall\nWk 17'] != 'ABS']  # drop ABS


## drop Nan for 2 old score ; merge to new column "BG_US"
df['BG_US_P'].fillna(0, inplace=True)
df['BG_US_A'].fillna(0, inplace=True)
df['BG_CH'].fillna(0, inplace=True)
##df['BG_US'] = df['BG_US_P'].astype(float) + df['BG_US_A'].astype(float) #--------------- sum of 2 columns






# replace
##df['BG_Cat'].replace('A','1', inplace=True)  # replace 1 item only
vals_to_replace = {'A':'1', 'B':'2', 'D':'3', 'S':'4', '':'0'}
s_to_replace = {'B':'1', 'G':'0'}
df['BG_Cat'] = df['BG_Cat'].map(vals_to_replace)
df['BG_Cat'] = df['BG_Cat'].astype(int)
df['BG_S'] = df['BG_S'].map(s_to_replace)
df['BG_S'] = df['BG_S'].astype(int)
df['BG_CH'] = df['BG_CH'].astype(int)
df['E_Total\nWk 17\n(60%)'] = df['E_Total\nWk 17\n(60%)'].astype(int)
df['Overall\nWk 17'] = df['Overall\nWk 17'].astype(float)
df['Overall\nWk 17'] = df['Overall\nWk 17'].round(0).astype(int)
df['BG_US_A'] = df['BG_US_A'].astype(float)
df['BG_US_A'] = df['BG_US_A'].round(0).astype(int)
df['BG_US_P'].astype(float)
df['BG_US_P'] = df['BG_US_P'].round(0).astype(int)


# select data | filter out Zero s
df_f = df[(df['BG_S'] == 0)]

# Export - before predict, check dataframe
df_f.to_csv('C:\\Users\\Administrator\\Desktop\\FYP\\Data\\Test1.csv', encoding='utf-8',index=False)

#'BG_Cat','BG_US_P','BG_CH','BG_Yr'
#'E_Total\nWk 17\n(60%)','Overall\nWk 17'
bg = ['BG_Cat','BG_US_P','BG_CH','BG_Yr']
# Predict from BG
X = df_f[bg]
y = df_f['Overall\nWk 17']
##y = df_f['E_Total\nWk 17\n(60%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.4)





########################################  Termination  #######################################################
##import sys
##sys.exit()
##df.columns.tolist()
# Compute the mean squared error of our predictions.
##mse = (((predictions - actual) ** 2).sum()) / len(predictions)


#------------------------------------------------------------------------------------------------------------
# LogReg
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X)

print('LogReg')
print(accuracy_score(y, y_pred_class))

df_f['Pred'] = y_pred_class


##S = set(y2) # collect unique label names
##D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
##Y = [D[y2_] for y2_ in y2] # store class labels as ints



# export for testing
df_f.to_csv('C:\\Users\\Administrator\\Desktop\\FYP\\Data\\Test.csv', encoding='utf-8',index=False)
df_f.to_excel(spreadsheet1)

# /LogReg
#-------------------------------------------------------------------------------------------------------------------------
### KNN
###10-fold cross-validation with K=5 for KNN (the n_neighbours parameter)
####knn = KNeighborsClassifier(n_neighbors=2)
####scores = cross_val_score(knn, X, y, cv=2, scoring='accuracy')
####print(scores)
####print(scores.mean())
##
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
k_range = range(1,24)
k_scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X)
    df_f['Pred_knn'] = predictions
    actual = y_test
    y_t= np.array(y).astype(int)
##print(accuracy_score(actual, predictions))
    print("KNN =",k)
    k_scores.append(accuracy_score(y_t, predictions.astype(int)))
    print(accuracy_score(y_t, predictions.astype(int)))
##mse = (((predictions - actual) ** 2).sum()) / len(predictions)
##    knn = KNeighborsClassifier(n_neighbors=k)
##    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
##    k_scores.append(scores.mean())
##print(k_scores)

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()



##n_neighbors = 5
##weights = 20
##
##knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
##    y_ = knn.fit(X, y).predict(T)
##
##df_f['Pred_KNN'] = regr_2.predict(X)

# export for testing
df_f.to_csv('C:\\Users\\Administrator\\Desktop\\FYP\\Data\\Test_KNN.csv', encoding='utf-8',index=False)
##df_f.to_excel(spreadsheet1)

# /KNN
#-------------------------------------------------------------------------------------------------------------------------
# Tree C4.5 | chi-squared 

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)

regr_1.fit(X_train, y_train)


# Predict
y_1 = regr_1.predict(X_test)

df_f['Pred_tree2'] = regr_1.predict(X)

y_test = np.array(y_test)

##S = set(y2) # collect unique label names
##D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
##Y = [D[y2_] for y2_ in y2] # store class labels as ints
print("depth = 2")
print(accuracy_score(y_test, y_1.astype(int)))
print(sklearn.metrics.explained_variance_score(y_test, y_1))
b=[]
for i in range(1,30):
    regr_2 = DecisionTreeRegressor(max_depth=i)
    regr_2.fit(X_train, y_train)
    y_2 = regr_2.predict(X_test)
    
    df_f['Pred_tree5'] = regr_2.predict(X)
    print("depth = ",i)
    print(accuracy_score(y_test, y_2.astype(int)))
    print(sklearn.metrics.explained_variance_score(y_test, y_2, multioutput='uniform_average'))
    b.append(sklearn.metrics.explained_variance_score(y_test, y_2, multioutput='uniform_average'))
a = range(1,30)
plt.plot(a, b)
plt.xlabel('Depth, for Decision Tree Regression')
plt.ylabel('explained_variance_score')
plt.show()
    

# plot graphviz   http://webgraphviz.com/
tree.export_graphviz(regr_1,out_file='tree2.dot')
tree.export_graphviz(regr_2,out_file='tree5.dot')


# export for testing
df_f.to_csv('C:\\Users\\Administrator\\Desktop\\FYP\\Data\\Test_tree.csv', encoding='utf-8',index=False)
df_f.to_excel(spreadsheet1)

# Tree
