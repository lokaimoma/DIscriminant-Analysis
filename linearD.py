from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()

# Convert dataset to pandas DataFrame
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

# Define predictor and response variables
X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['species']

# Fit the LDA model
model = LinearDiscriminantAnalysis()
model.fit(X, y)

# Define method to evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean Accuracy:", np.mean(scores))

# Define new observation
new = np.array([[5, 3, 1, 0.4]])

# Predict which class the new observation belongs to
predicted_class = model.predict(new)


print("Predicted class:", predicted_class)

# # Define data to plot
# data_plot = model.fit(X, y).transform(X)
#
# # Create LDA plot
# plt.figure()
# colors = ['red', 'green', 'blue']
# lw = 2
# for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
#     plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
#                 label=target_name)
#
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.show()
