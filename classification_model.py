import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as tts,cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import pylab as pl
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors

#reading the dataframe
df=pd.read_table('final_events/final_peaks_updated_new.txt',index_col=0,sep=',')

# cols=[str(i) for i in range(70)]
# cols+=['labels','phase']

#for i in range(7):
#df.columns=cols
#finding the single event devices
freq=df.groupby('labels').count()
single_freq=list(freq[freq['0']==1].index)
print ()
df=df[df['phase']=='B']
del df['phase']

#eliminating all the single event devices
for i in single_freq:
    df=(df[df['labels']!=i])

print("Events left after removal of single events %d"%len(df))

#df=_df


device_list=df['labels'].unique()
print('output devices %s'%len(device_list))

#Cross Validation
# # clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
# # clf = clf.fit(feature_train,label_train)
# # result = clf.predict(feature_test)
# # accuracy_score(label_test,result)
# # print (classification_report(label_test,result,digits=4))

# # scores = cross_val_score(clf, feature_matrix, labels)
# # scores.mean()  
# # clf = ExtraTreesClassifier(n_estimators=150)
# # scores = cross_val_score(clf, feature_matrix, labels, cv=10)
# # scores.mean()
# # clf = clf.fit(feature_train,label_train)
# clf = svm.SVC(C=1.0,kernel='rbf',cache_size=1000,decision_function_shape='ovr',shrinking=True,probability=True)
# scores = cross_val_score(clf,feature_matrix,labels,cv=StratifiedKFold(n_splits=4,shuffle=True))
# print (scores, scores.mean())
# clf.fit(feature_train, label_train)

'''Extra-Trees'''
clf = ExtraTreesClassifier(n_estimators=200,n_jobs=-1,max_features=30,criterion='gini')
scores = cross_val_score(clf,feature_matrix,labels,cv=StratifiedKFold(n_splits=4,shuffle=True))
print (scores, scores.mean())
clf = clf.fit(feature_train,label_train)
result = clf.predict(feature_test)
accuracy_score(label_test,result)
print (classification_report(label_test,result,digits=4))
print (clf.max_depth)
clf.get_params()# print(classification_report_imbalanced(label_test, result))
clf.score(feature_test,label_test)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print ('hlo',clf.oob_score_)

cm=sklearn.metrics.confusion_matrix(label_test,result )
print(cm)
pl.matshow(cm)
pl.colorbar()
pl.show()

