import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


file_name = 'DecisionTree.pkl'

# Loading pickle model to predict data from test_data.csv file
pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)

dataSet_testdata = pd.read_csv('test_data.csv')

x1_index = dataSet_testdata.columns.get_loc("Age")
y_index = dataSet_testdata.columns.get_loc("EstimatedSalary")

x_testdata = dataSet_testdata.iloc[:, [x1_index]]
y_testdata = dataSet_testdata.iloc[:, y_index:(y_index+1)]

y_pred = model_pkl.predict(x_testdata)

print(y_pred)