from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import datasets
import pandas as pd
import numpy as np

# Load iris dataset
iris = datasets.load_iris()

# Convert dataset to pandas DataFrame
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

# Define predictor and response variables
X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['target']
# print(df)
# Fit the QDA model
model = QuadraticDiscriminantAnalysis()
model.fit(X, y)

# Define method to evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluate model using cross-validation
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Mean scores:", np.mean(scores))

# Define new observation
new = np.array([[5, 3, 1, 0.4]])  # Adjusted to a 2D array

# Predict which class the new observation belongs to
prediction = model.predict(new)
prediction_class = iris.target_names[int(prediction)]
print("Predicted class:", prediction_class)