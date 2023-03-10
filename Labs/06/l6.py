import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import   KNeighborsClassifier
from sklearn.linear_model import LogisticRegression       
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve

iris_df = datasets.load_iris()

# Using K-means
model = KMeans(n_clusters=3)

# Fitting the k-means
model.fit(iris_df.data)

# Predicitng
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])
all_predictions = model.predict(iris_df.data)

print(predicted_label)
print(all_predictions)

# Using seeds dataset
seeds_df = pd.read_csv(
"https://raw.githubusercontent.com/vihar/unsupervised-learning-with-python/master/seeds-less-rows.csv")

varieties = list(seeds_df.pop('grain_variety'))

samples = seeds_df.values

# Using Complete Linkage
mergings = linkage(samples, method='complete')

dendrogram(mergings,
labels=varieties,
leaf_rotation=90,
leaf_font_size=6,
)

plt.show()

y = df['target'].values      
X = df.drop('target', axis=1).values

# Using kNN classifier    
knn = KNeighborsClassifier(n_neighbors =   6)

# Fit the classifier to the data       
knn.fit(X,y)                    
new_prediction = knn.predict(X_new)
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop
for i, k in enumerate(neighbors):   
   knn = KNeighborsClassifier(n_neighbors=k)
   
# Fitting the classifier
   knn.fit(X_train, y_train)
   
# Cecking accuracy
   train_accuracy[i] = knn.score(X_train, y_train)   
   test_accuracy[i] = knn.score(X_test, y_test)
   
# Plotting
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')

# Other method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)              
# Classifiers  
logreg = LogisticRegression()

# Fitting the classifier     
logreg.fit(X_train,y_train)                  
y_pred = logreg.predict(X_test) 
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generating curve values   
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plotting      
plt.plot([0, 1], [0, 1], 'k')       
plt.plot(fpr, tpr)       
plt.xlabel('False Positive')       
plt.ylabel('True Positive')       
plt.title('Curve')       
plt.show()
