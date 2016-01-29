import numpy
import random
import math
from sklearn import linear_model,cross_validation,preprocessing
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
import entropy
import pca2
import timeit
import matplotlib
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
import matplotlib.pyplot as plt
dataset="none"


dataset="mfeat"
path=dataset+'/'
readdata = numpy.loadtxt(path+'all_mfeat.csv',delimiter=',')
#q=readdata.shape[1]
train_data, test_data, y_train, y_test = cross_validation.train_test_split(readdata, readdata[:,0], test_size=0.5, random_state=0)
mylambda=100000.0
#C=1/alpha

n=train_data.shape[1]
m=test_data.shape[0]
#print train_data.shape[0],n, m 

print 'Dataset:',dataset
valid_data, test_data, y_valid, y_test = cross_validation.train_test_split(test_data, test_data[:,0], test_size=0.25, random_state=1)

## fetch labels from dataset###
y_train=train_data[:,0]
#y_valid=test_data[:m/2,0]
#y_test=test_data[m/2:,0]

#print y_train
#print y_test

basen=5
#random.seed(1234)
#list_rand = range(1,5)
#list_rand=random.sample(range(1,n), basen)

figpath=path+'fig/'+dataset

#list_rand = range(17,n)
#figpath=path+'/fig_c/'+dataset



print '# Features:',n
print '# Base Features:',basen
filename_init_attr=path+dataset+'.init'
#list_rand=random.sample(range(1,n), basen)
#list_rand=numpy.loadtxt(filename_init_attr,delimiter=',')
#list_rand = numpy.array(list_rand, int).tolist()     
#numpy.savetxt(filename_init_attr,list_rand,delimiter=',')
list_rand = list(numpy.loadtxt(filename_init_attr,delimiter=','))
#print list_rand
list_base = list_rand*1
basen =len(list_rand)

list_inc=[]
list_exc=[]
change=[]
changeacc=[]
ig=[]
correlation=[]
del_auc=[]
del_acc=[]
new_correlation=[]
clf = linear_model.LogisticRegression(C=mylambda, class_weight='auto')
c=[0,0,0,0]

x_train=train_data[:,list_rand]
x_valid=valid_data[:,list_rand]
x_test=test_data[:,list_rand]
clf.fit(x_train,y_train)
#dlabel=clf.predict(x_test) 
 
dec_val_test=[]
"""
for i in range(test_data.shape[0]):
 dec_val_test.append(clf.decision_function(x_test[i,:])[0])
"""
#ptest=clf.predict_proba(x_test)
#dec_val_test=ptest[:,1]
dec_val_test=clf.decision_function(x_test)
auc_base=roc_auc_score(y_test,dec_val_test)

pred_val_test=clf.predict(x_test) 
acc_base=accuracy_score(y_test,pred_val_test)
while auc_base>0.7:
   print auc_base
   list_rand=random.sample(range(1,n), basen)
   list_base = list_rand*1
   x_train=train_data[:,list_base]
   x_valid=valid_data[:,list_base]
   x_test=test_data[:,list_base]
   clf.fit(x_train,y_train)
   dec_val_test_base=clf.decision_function(x_test)
   auc_base=roc_auc_score(y_test,dec_val_test_base)
   pred_val_test=clf.predict(x_test) 
   acc_base=accuracy_score(y_test,pred_val_test)
   numpy.savetxt(filename_init_attr,list_rand,delimiter=',')

print "base:",auc_base
ent=[]
ent_auc=[]
auc_bool=[]
clf = linear_model.LogisticRegression(C=10000, class_weight='auto')
#clf2 = linear_model.LogisticRegression(C=0.01, penalty='l1',class_weight='auto')
#list_exc = list(set(range(1,n)) - set(list_base))\
ranked_feat = numpy.loadtxt(path+'ent',delimiter=',')
print ranked_feat
top = 0
for i in range(0,ranked_feat.shape[0]):
   if ranked_feat[i,1] >=0:
      top= i
      break;
print top
list_exc = ranked_feat[0:top,0]
list_exc = numpy.array([int(x) for x in list_exc])
gene_c=0
end= 0
print len(list_exc)
steps = range(0,len(list_exc),20)
print steps
feature_ids=[]
auc=[]
for i in range(0,len(steps)-1):
   begin = steps[i]
   end = steps[i+1]
   feature_ids.extend(list_exc[range(begin,end)])
   x_train=train_data[:,feature_ids]
   x_valid=valid_data[:,feature_ids]
   x_test=test_data[:,feature_ids]
   clf.fit(x_train,y_train)
   dec_val_test=clf.decision_function(x_test)
   auc.append(roc_auc_score(y_test,dec_val_test))
   #print feature_ids
feature_ids.extend(list_exc[range(begin,end)])
#print feature_ids
x_train=train_data[:,feature_ids]
x_valid=valid_data[:,feature_ids]
x_test=test_data[:,feature_ids]
clf.fit(x_train,y_train)
dec_val_test=clf.decision_function(x_test)
auc.append(roc_auc_score(y_test,dec_val_test))
clf2 = linear_model.LogisticRegression(C=0.01, penalty='l1')
clf2.fit(x_train,y_train)
dec_val_test=clf2.decision_function(x_test)
auc_lasso = roc_auc_score(y_test,dec_val_test)
lasso =[auc_lasso] * len(list_exc)
#plt.title(",fontsize=18)
plt.xlabel("Batch features",fontsize=16)
plt.ylabel("AUC",fontsize=16)
#s
plt.plot(steps, auc,linewidth=0.5,  marker='s',mfc='k',color='k',label="Proposed")
plt.plot(range(0,len(list_exc)), lasso,linewidth=0.5,  ls='dashed',color='k',label="Lasso")
plt.xticks(steps)
plt.ylim(ymax=1.0001)
plt.legend(loc=4)
plt.savefig(figpath+'compare.eps', format='eps', dpi=1200)
plt.show()
