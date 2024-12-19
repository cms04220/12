#1
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

filename = "./data/1_pima.csv"

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#2
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#3
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

y_pred_binary= (y_pred > 0.5).astype(int)
print(y_pred_binary)

accuracy = accuracy_score(y_test, y_pred_binary)

#4
print("------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred_binary)
print("------------------------")
print("Accuracy:", accuracy)

#5
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='predicted')

plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Data Index')
plt.ylabel('Class (0 or 1)')
plt.legend()

plt.savefig("./results2/linear_regression.png")

#6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

filename = "./data/1_pima.csv"

column_names = ['pred', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#7
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=41)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("-------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("-------------------------")

#8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

filename = "./data/1_pima.csv"

column_names = ['preg', 'plas' ,'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#9
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=41)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("-------------------------")
print("Actual Values", y_test)
print("Predicted Values:", y_pred)
print("-------------------------")
print("Accuracy:", accuracy)

#10
model = DecisionTreeClassifier(max_depth=1000, min_samples_split=60, min_samples_leaf=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("-------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("-------------------------")
print("Accuracy", accuracy)

