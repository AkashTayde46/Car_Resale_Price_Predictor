import pandas as pd;
import numpy as np;
from sklearn.ensemble import RandomForestRegressor
import pickle; 
df = pd.read_csv('car_resale_prices.csv')
X = np.array(df.iloc[:, 1:9])
y = np.array(df.iloc[:, 0:1])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model1 = RandomForestRegressor()
model1.fit(X_train, y_train)

pickle.dump(model1, open('final_model.pkl','wb'))
model = pickle.load(open('final_model.pkl','rb'))