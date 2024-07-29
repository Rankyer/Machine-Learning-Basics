import numpy as np                                                              
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

tennis_data = pd.read_csv('tennis.csv', sep='\s*,\s*', engine = 'python')
o_l = LabelEncoder()
t_l = LabelEncoder()
h_l = LabelEncoder()
w_l = LabelEncoder()
d_l = LabelEncoder()

t_outlook = o_l.fit_transform(tennis_data['Outlook'])
t_temp = t_l.fit_transform(tennis_data['Temp'])
t_humid = h_l.fit_transform(tennis_data['Humidity'])
t_wind = w_l.fit_transform(tennis_data['Wind'])
t_decision = d_l.fit_transform(tennis_data['Decision'])
print(t_outlook)

labels = np.concatenate((t_outlook.reshape(-1, 1),
t_temp.reshape(-1, 1),
t_humid.reshape(-1, 1),
t_wind.reshape(-1, 1)), axis = 1)
targets = t_decision.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(labels, targets,
test_size = 5)
print("Training input:")
print(X_train)
print("Training targets:")
print(Y_train)

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
train_predict = clf.predict(X_train).reshape(-1,1)
test_predict = clf.predict(X_test).reshape(-1,1)

train_perf = accuracy_score(Y_train, train_predict)
test_perf = accuracy_score(Y_test, test_predict)

print("Train accuracy: %3.2f, Test accuracy: %3.2f" % (train_perf, test_perf))
X_trans = X_test.transpose()

X_labels = [o_l.inverse_transform(X_trans[0])]
X_labels.append(t_l.inverse_transform(X_trans[1]))
X_labels.append(h_l.inverse_transform(X_trans[2]))
X_labels.append(w_l.inverse_transform(X_trans[3]))

X_labels.append(d_l.inverse_transform(np.ravel(test_predict)))
X_labels.append(d_l.inverse_transform(np.ravel(Y_test)))

# Tranpose it back to num_samples x num_columns
results = np.array(X_labels).transpose()

for cname in tennis_data.columns:
    print(cname + '\t', end = '')

print("Predicted\tActual")
print("----------------------------------------------------------------------------------------------\n")

for row in results:
    for col in row:
        print("%s\t\t" % col, end = '')
    print()
print()                 