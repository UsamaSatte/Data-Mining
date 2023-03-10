import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data
Y = iris.target

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Usama Mahmood (0710399)')
plt.savefig('iris_scatter.png')
plt.show()

plt.hist(X[:, 0], bins=10)
plt.xlabel('Sepal length')
plt.ylabel('Frequency')
plt.title('Usama Mahmood (0710399)')
plt.savefig('iris_hist.png')
plt.show()
