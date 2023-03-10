# Import the required library

from sklearn.datasets import load_boston

import pandas as pd

# Load the dataset

X, Y = load_boston(return_X_y=True)

# Convert the matrix to dataframe

Xdf = pd.DataFrame(X)

Ydf = pd.DataFrame()

Ydf['Y'] = Y

print('Sample data : \n',Xdf.iloc[:10])

# Check the type of each column

print('Datatype of each column : \n',Xdf.dtypes)

# Discretize target column

print(pd.cut(Ydf.Y, labels = [1,2,3], bins=3))