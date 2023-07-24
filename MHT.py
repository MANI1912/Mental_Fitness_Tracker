import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
#import kaleido.scopes.plotly
import plotly.offline as pyo
#from kaleido.scopes.plotly import PlotlyScope
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
df1 = pd.read_csv("C:\\Users\\vsrm\\OneDrive\\Desktop\\IBM_Skills-Build_AI_Internship-master\\mental-and-substance-use-as-share-of-disease.csv")
df2 = pd.read_csv("C:\\Users\\vsrm\\OneDrive\\Desktop\\IBM_Skills-Build_AI_Internship-master\\prevalence-by-mental-and-substance-use-disorder.csv")
df1.head()
df2.head()
data = pd.merge(df2, df1)
data.head()
data.isnull().sum()
data.drop('Code', axis = 1, inplace = True)
data.size
data.shape
data = data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety_disorder','Drug_usage_disorder','Depression_disorder','Alcohol_use_disorder','Mental_fitness'], axis = 1)
data.head()
min1 = data['Mental_fitness'].min()
max1 = data['Mental_fitness'].max()
print(data.loc[data['Mental_fitness'] == min1, 'Country'].values[0])
print(data.loc[data['Mental_fitness'] == max1, 'Country'].values[0])
data.groupby('Country')['Mental_fitness'].mean()
data.info()
data.describe()
plt.figure(figsize=(12,8))
sns.heatmap(data.drop('Country', axis = 1).corr(),annot=True,cmap='Blues')
plt.show()
sns.jointplot(x = 'Schizophrenia',y = 'Mental_fitness',data = data, kind = 'reg', color='g')
plt.show()
sns.jointplot(x = 'Bipolar_disorder', y = 'Mental_fitness', data = data, kind='reg', color='b')
plt.show()
sns.jointplot(x = 'Depression_disorder', y = 'Mental_fitness', data = data, kind='reg', color='m')
plt.show()
sns.jointplot(x = 'Drug_usage_disorder', y = 'Mental_fitness', data = data, kind='reg', color='r')
plt.show()
sns.pairplot(data = data, corner = True)
plt.show()
fig = px.pie(data, values='Mental_fitness', names='Year')
fig.show()
#kaleido_scope = kaleido.scopes.plotly.PlotlyScope(mathjax=None)

# Save the plot as an image
#pio.write_image(fig, "pie1.png", engine = "kaleido")
#fig = px.line(data[data['Country'].isin(["India", "South Africa", "United States", "United Kingdom", "Japan", "China", "United Arab Emirates", "Australia"])], x="Year", y="Mental_fitness", color='Country', markers=True, color_discrete_sequence=['black','red', 'green', 'blue','orange', 'purple', 'violet', 'steelblue'], template = 'plotly')
# fig.show()
#kaleido_scope = kaleido.scopes.plotly.PlotlyScope(mathjax=None)

# Save the plot as an image
#pio.write_image(fig, "line.png", engine = "kaleido")
l = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = l.fit_transform(data[i])
data
X = data.drop('Mental_fitness', axis = 1)
Y = data['Mental_fitness']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred_train = rf.predict(x_train)
plt.figure(figsize = (12, 8))
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
plt.xlabel('Actual Values (y_train)')
plt.ylabel('Predicted Values (Y_pred)')
plt.title('Actual vs. Predicted Values')
plt.show()
plt.figure(figsize = (12, 8))
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual Values')
plt.scatter(range(len(y_pred_train)), y_pred_train, color='red', label='Predicted Values')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()
#print("The model performance for training set")
#print("--------------------------------------")
#print('MSE is {}'.format(mse_train))
#print('RMSE is {}'.format(rmse_train))
#print('R2 score is {}'.format(r2_train))
#print("\n")
y_pred_test = rf.predict(x_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("The model performance for testing set")
print("-------------------------------------")
print('MSE is {}'.format(mse_test))
print('RMSE is {}'.format(rmse_test))
print('R2 score is {}'.format(r2_test))
print("\n")
plt.figure(figsize = (12, 8))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (Y_pred_test)')
plt.title('Actual vs. Predicted Values')
plt.show()
plt.figure(figsize = (12, 8))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
plt.scatter(range(len(y_pred_test)), y_pred_test, color='red', label='Predicted Values')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()
