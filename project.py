import pandas as pd
fish = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Fish.csv')


fish.head()
fish.info()


fish.describe()
fish.columns
y = fish['Weight']
X = fish[['Category','Height', 'Width', 'Length1',
       'Length2', 'Length3']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)