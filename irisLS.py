import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('iris.csv')

feature_pairs = [('sepal_length', 'sepal_width'), ('sepal_length', 'petal_length'), ('sepal_length', 'petal_width'),
                 ('sepal_width', 'petal_length'), ('sepal_width', 'petal_width'), ('petal_length', 'petal_width')]

plt.figure(figsize=(15, 10))

le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])

for i, (feature_x, feature_y) in enumerate(feature_pairs, 1):
    X = data[[feature_x, feature_y]]
    y = data['species_encoded']

    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # Plot the data points
    plt.subplot(2, 3, i)
    plt.scatter(X[feature_x], X[feature_y], c=y, cmap='viridis')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'{feature_x} vs {feature_y}')

    # Plot the decision boundaries
    xlim = plt.xlim()
    ylim = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)

plt.tight_layout()
plt.show()
