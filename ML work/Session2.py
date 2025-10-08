import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\wifi\Desktop\Depi_Amit_BNS3\Depi_Amit_AI_BNS3\ML work\Salary_Data.csv")
data.head()
#plt.figure(figsize=(10, 6))
#sns.pairplot(data, x_vars=['YearsExperience'], y_vars='Salary', height=6, aspect=1, kind='scatter')
#plt.title('Years of Experience vs Salary')
#plt.show()
#x = data.iloc[:, :-1].values
#y = data.iloc[:, 1].values
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)   
#my_model = LinearRegression()
#my_model.fit(x_train, y_train)
#plt.scatter(x_train, y_train, color='blue')
#plt.plot(x_train, my_model.predict(x_train), color='red')
#plt.title('Years of Experience vs Salary (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()
