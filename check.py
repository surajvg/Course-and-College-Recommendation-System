import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
#from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

# Setting plot style
plt.style.use('ggplot')

#cities = pd.DataFrame([['Sacramento', 'California']], columns=['Name', 'Age','C_education','Grade','AreaofIntrest','Skills'])
#cities.to_csv('course_data1.csv')

course_df = pd.read_csv("course_data1.csv", encoding= 'unicode_escape')
user_df = pd.read_csv("course_data.csv", encoding= 'unicode_escape')

print(user_df.head())

X = course_df.drop(columns='age')
y = course_df["age"].values  
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=101)

course_df = X_train.copy()
course_df["age"] = y_train

#course_df


baseline_y_pred = pd.DataFrame(course_df.groupby('C_education')['age'].mean())
val_course_dict = {'C_education': X_val["C_education"], 'Actual age': y_val}  
val_course_df = pd.DataFrame(val_course_dict)


y_pred_and_y_true = pd.merge(baseline_y_pred, val_course_df, on='C_education')
baseline_y_pred_vs_y_true = y_pred_and_y_true.rename(columns={"age": "Predicted age"})

#baseline_y_pred_vs_y_true
#Root Mean Square Error
#print("RMSE baseline model: ", sqrt(mean_squared_error(baseline_y_pred_vs_y_true["Actual age"], 
#                                                       baseline_y_pred_vs_y_true["Predicted age"])))
content_train_df = pd.merge(course_df, user_df, on=['age','C_education','area','interests'])

#content_train_df
y_grouped_by_user = content_train_df.groupby(["studylevel"])
y_train_listed = []

for i, j in y_grouped_by_user:
    y_train_listed.append(j["Course"].values)  
    
y_train_listed[0]