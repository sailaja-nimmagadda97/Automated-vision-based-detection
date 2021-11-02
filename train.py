# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import cv2
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
# load the user configs
with open('conf/conf.json') as f:
	config = json.load(f)

# config variables
test_size 		= config["test_size"]
seed 			= config["seed"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
results 		= config["results"]
classifier_path = config["classifier_path"]
train_path 		= config["train_path"]
num_classes 	= config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)
print(labels_string[0])
h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ((trainData.shape)[0])
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print ("[INFO] creating model...")
#model = DecisionTreeClassifier(criterion="entropy",max_depth=1)
model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
#print(trainData)
#model.fit(trainData, trainLabels)
scores = cross_val_score(model,trainData, trainLabels, cv=6)
print (scores)
#kf = KFold(n_splits=6)
#kf.get_n_splits(trainData) # returns the number of splitting iterations in the cross-validator

#print(kf)
#k-fold cross validation
mean_auc = 0.0
n = 6  # repeat the CV procedure 10 times to get more precise results
for i in range(n):
	X_train, X_cv, y_train, y_cv = train_test_split(np.array(trainData), np.array(trainLabels), test_size=.20 , random_state=i*9)
	model.fit(X_train, y_train)
	roc=0
	#h=(X_cv.shape)
	print((X_cv.shape)[0])
	#g=format(y_cv.shape)
	#print("[INFO] labels shape: {}".g[0])
	for j in range((X_cv.shape)[0]):
		preds = model.predict_proba((X_cv))[j]
		preds= np.argsort(preds)[::-1][:5]
		#print(preds)
		#print(y_cv[j])
		if y_cv[j]==preds[0]:
			roc+= 1
	roc = (roc / float(len(y_cv))) * 100
	print("roc: {:.2f}%\n".format(roc))
	mean_auc+=roc
#accuracies
print ("Mean AUC:{:.2f}%\n". format(mean_auc/n))
print ("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0
train_labels = os.listdir(train_path)
# loop over test data
for (label, features) in zip(testLabels, testData):
	# predict the probability of each class label and
	# take the top-5 class labels
	predictions = model.predict_proba(np.atleast_2d(features))[0]
	predictions=np.argsort(predictions)[::-1][:5]


	# rank-1 prediction increment
	#print(train_labels[label])
	#print(train_labels[predictions[0]])

	if label == predictions[0]:
		rank_1 += 1

	# rank-5 prediction increment
	if label in predictions:
		rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100
print("Rank-1: {:.2f}%\n".format(rank_1))
print("Rank-5: {:.2f}%\n".format(rank_5))
# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()
