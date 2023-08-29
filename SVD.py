from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('fashion-mnist_train.csv')
df2 = pd.read_csv('fashion-mnist_test.csv')

train_data = df.to_numpy()
test_data = df2.to_numpy()

# Separate the labels from the data
train_unlabeled = train_data[:48000, 1:]
train_labels = train_data[:48000, 0]
val_unlabeled = train_data[48000:, 1:]
val_labels = train_data[48000:, 0]
test_unlabeled = test_data[:, 1:]
test_labels = test_data[:, 0]

score = 0  # Max accuracy score for KNN validation
k_val = 1  # Initial k value for KNN
SS = StandardScaler()
TSVD = TruncatedSVD(n_components=4)  # Number of Dimensions
GNB = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=k_val)
MLR = LogisticRegression(solver='saga', max_iter=1000)
GNB2 = GaussianNB()
KNN2 = KNeighborsClassifier(n_neighbors=k_val)
MLR2 = LogisticRegression(solver='saga', max_iter=1000)

# Scale the unlabeled data (training and test)
train_scaled = SS.fit_transform(train_unlabeled)
val_scaled = SS.fit_transform(val_unlabeled)
test_scaled = SS.fit_transform(test_unlabeled)

# Perform SVD on the unlabeled data (training and test)
reduced_train = TSVD.fit_transform(train_scaled)
reduced_val = TSVD.fit_transform(val_scaled)
reduced_test = TSVD.fit_transform(test_scaled)

# print(np.shape(reduced_val))

print(TSVD.explained_variance_ratio_ * 100)
print(TSVD.explained_variance_.sum() / 4)

# Train models with dimensionally reduced training set
GNB.fit(reduced_train, train_labels)
KNN.fit(reduced_train, train_labels)
MLR.fit(reduced_train, train_labels)

# print(k_val)
# Tune the k-value for KNN
for i in range(1, 50, 2):
    KNN.set_params(n_neighbors=i)
    # KNN.predict(reduced_val)
    val_score = KNN.score(reduced_val, val_labels)
    if val_score > score:
        score = val_score
        k_val = i

KNN.set_params(n_neighbors=k_val)
# print(k_val)

#  Test on dimensionally reduced test set (kept for ease of access despite it being unnecessary)
# GNB_pred = GNB.predict(reduced_test)
# KNN_pred = KNN.predict(reduced_test)
# MLR_pred = MLR.predict(reduced_test)

# Tests the models on the reduced test set and prints the accuracy
print('NB: ', GNB.score(reduced_test, test_labels))
print('KNN: ', KNN.score(reduced_test, test_labels))
print('MLR: ', MLR.score(reduced_test, test_labels))

# Train models on original training set
GNB2.fit(train_unlabeled, train_labels)
KNN2.fit(train_unlabeled, train_labels)
# MLR2.fit(train_unlabeled, train_labels)

score = 0
k_val = 1
# print(k_val)
# Tune the k-value for KNN2
for i in range(1, 10, 2):
    KNN2.set_params(n_neighbors=i)
    val_score = KNN2.score(val_unlabeled, val_labels)
    if val_score > score:
        score = val_score
        k_val = i

KNN2.set_params(n_neighbors=k_val)
# print(k_val)

# GNB2_pred = GNB2.predict(test_unlabeled)
# KNN2_pred = KNN2.predict(test_unlabeled)
# MLR2_pred = MLR2.predict(test_unlabeled)

# Tests the models on the original test set and prints the accuracy
print('NB2: ', GNB2.score(test_unlabeled, test_labels))
print('KNN2: ', KNN2.score(test_unlabeled, test_labels))
# print('MLR2: ', MLR2.score(test_unlabeled, test_labels))
