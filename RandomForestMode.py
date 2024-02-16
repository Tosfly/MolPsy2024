#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:22:54 2020

@author: Dr. yi-zhi wang
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##Thank you, Sreenivas Bhattiprolu, for the original codes.

############################################ UPSAMPLING RANDOM FOREST ################################################### 
# #For roc_auc_score in the multiclass case, these must be probability estimates which sum to 1.
import os
os.chdir("/Users/yi-zhiwang/Projects/BirA/Raw Data/TMTMS/RandomForrest")

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#Read data ready for ML analysis
df = pd.read_csv("A2a_BR1_ML.csv")
print(df.head())

#Count how many true postsynaptic proteins 
sizes = df['Class'].value_counts(sort = 1)

from sklearn.utils import resample
print(df['Class'].value_counts())


#Separate majority and minority classes
df_important = df[df['Class'] == 1]
df_majority = df[df['Class'] == 0]
df_minority = df[df['Class'] == 1]

# Upsample minority class and other classes separately
# If not, random samples from combined classes will be duplicated and we run into
#same issue as before, undersampled remians undersampled.
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=619,    # to match average class
                                 random_state=42) # reproducible results
 
df_important_upsampled = resample(df_important, 
                                 replace=True,     # sample with replacement
                                 n_samples=619,    # to match average class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_important_upsampled, df_minority_upsampled])
print(df_upsampled['Class'].value_counts())

Y_upsampled = df_upsampled["Class"].values

#Define the independent variables
X_upsampled = df_upsampled.drop(labels = ['Locus','FullName','Abbre','Class'], axis=1) 

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled = train_test_split(X_upsampled, 
                                                                                            Y_upsampled, 
                                                                                            test_size=0.15, 
                                                                                            random_state=20)#20 or 42


#Train again with new upsamples data
from sklearn.ensemble import RandomForestClassifier
model_RF_upsampled = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Train the model on training data
model_RF_upsampled.fit(X_train_upsampled, y_train_upsampled)
prediction_test_RF_upsampled = model_RF_upsampled.predict(X_test_upsampled)

from sklearn import metrics
print("********* METRICS FOR BALANCED DATA USING UPSAMPLING *********")
print ("Accuracy = ", metrics.accuracy_score(y_test_upsampled, prediction_test_RF_upsampled))

from sklearn.metrics import confusion_matrix
cm_upsampled = confusion_matrix(y_test_upsampled, prediction_test_RF_upsampled)
print(cm_upsampled)

feature_list = list(X_upsampled.columns)
feature_imp = pd.Series(model_RF_upsampled.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

############################################ Evaluation of MODEL BY ROC ################################################
# #Right metric is ROC AUC
# #Starting version 0.23.1 you can report this for multilabel problems. 
# #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
from sklearn.metrics import roc_auc_score  #Version 0.23.1 of sklearn
prob_y_test_upsampled = model_RF_upsampled.predict_proba(X_test_upsampled)[:,1]
# print("ROC_AUC score for imbalanced data is:")
roc_auc_score(y_test_upsampled, prob_y_test_upsampled)

from sklearn.metrics import plot_roc_curve

ax = plt.gca()
model_RF_disp = plot_roc_curve(model_RF_upsampled, X_test_upsampled, y_test_upsampled, ax=ax, alpha=0.8)
##plt.show()


############################################ SAVE MODEL/PREDICTION ####################################################
#X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels = ['Locus','FullName','Abbre','Class'], axis=1) 
print(X.head())

NewSynPredict = model_RF_upsampled.predict(X)
#Add a new column to original dataframe
df.insert(12, 'Prediction', NewSynPredict, True) 
#export data as csv
df.to_csv('Prediction.csv',index=False)
























