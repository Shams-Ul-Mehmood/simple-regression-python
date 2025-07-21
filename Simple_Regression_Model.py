import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("E:\\Cybersecurity\\3rd Semester\\ML\\Labs\\products.csv")   # Edit the file path.
product_data = pd.DataFrame(data)
product_name = np.array(product_data.Name)
product_price = np.array(product_data.Price)
plt.scatter(product_name, product_price)
plt.xlabel("Product Name")
plt.ylabel("Product Price")
plt.show()

product_brand = np.array(product_data.Brand)
plt.scatter(product_brand, product_price)
plt.xlabel("Product Brand")
plt.ylabel("Product Price")
plt.show()

product_Size = np.array(product_data.Size)
plt.scatter(product_Size, product_price)
plt.xlabel("Product Size")
plt.ylabel("Product Price")
plt.show()

numeric_data = product_data.select_dtypes(include=['number'])
correlation = numeric_data.fillna(0).corr()
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, cmap="Blues")

le = LabelEncoder()
for col in ['name', 'brand', 'size']:
    data[col] = le.fit_transform(data[col].astype(str))

# df_encoded = pd.get_dummies(product_data[['name', 'brand', 'size']], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(data[['name', 'brand', 'size']], data['price'], test_size=0.3, random_state=41)
# x_train, x_test, y_train, y_test = train_test_split(product_data[["name","brand","size"]], product_data.price, test_size=0.3, random_state=41)

regression = linear_model.LinearRegression()
regression.fit(x_train, y_train)
print(regression.coef_)
print(regression.intercept_)

y_predicted = regression.predict(x_test)

x = np.array(x_test.name)
y = np.array(y_test)
predicted_y = np.array(y_predicted)
plt.scatter(x, y)
plt.scatter(x, predicted_y, color="Pink", linewidths=2)
plt.show()
