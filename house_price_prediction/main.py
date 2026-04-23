import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "square_feet": [1500, 1800, 2400, 3000, 3500],
    "bedrooms": [3, 4, 3, 5, 4],
    "bathrooms": [2, 3, 2, 4, 3],
    "price": [300000, 400000, 500000, 600000, 650000]
}

df = pd.DataFrame(data)

X = df[["square_feet", "bedrooms", "bathrooms"]]
y = df["price"]

model = LinearRegression()

model.fit(X, y)

new_house = np.array([[2000, 3, 2]])
predicted_price = model.predict(new_house)

print("Predicted Price:", predicted_price[0])

plt.scatter(df["square_feet"], df["price"], color="blue")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("House Price vs Square Feet")
plt.show()