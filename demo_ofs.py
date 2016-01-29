import numpy
import random
import math
import sys
from sklearn import linear_model,cross_validation,preprocessing
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
import scipy.stats as stats
import entropy
import pca2
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
print train_data.shape[0],n, m 

print 'Dataset:',dataset
valid_data, test_data, y_valid, y_test = cross_validation.train_test_split(test_data, test_data[:,0], test_size=0.25, random_state=1)

## fetch labels from dataset###
y_train=train_data[:,0]



#start with a very few features to build base model
basen=5
#random.seed(1234)
#list_rand = range(1,5)
list_rand=random.sample(range(1,n), basen)
figpath=path+'fig/'+dataset

#list_rand = range(17,n)
#figpath=path+'/fig_c/'+dataset

basen =len(list_rand)

print '# Features:',n
print '# base features:',basen
filename_init_attr=path+dataset+'.init'
#list_rand=random.sample(range(1,n), basen)
#list_rand=numpy.loadtxt(filename_init_attr,delimiter=',')
#list_rand = numpy.array(list_rand, int).tolist()     
numpy.savetxt(filename_init_attr,list_rand,delimiter=',')
list_base = list_rand*1

list_inc=[]
list_exc=[]
change=[]
changeacc=[]
ig=[]
del_auc=[]
del_acc=[]
clf = linear_model.LogisticRegression(C=mylambda, class_weight='auto')
c=[0,0,0,0]

x_train=train_data[:,list_rand]
x_valid=valid_data[:,list_rand]
x_test=test_data[:,list_rand]
clf.fit(x_train,y_train)
#dlabel=clf.predict(x_test) 
 
dec_val_test=[]

dec_val_test=clf.decision_function(x_test)
auc_base=roc_auc_score(y_test,dec_val_test)

pred_val_test=clf.predict(x_test) 
acc_base=accuracy_score(y_test,pred_val_test)
#build base model such that the AUC is not already very high by chance
while auc_base>0.7:
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

list_exc = list(set(range(1,n)) - set(list_base))
gene_c=0
for gene in list_exc:#build set for every gene
   sys.stdout.write('Evaluating Feature ID: %s\r' % gene)
   sys.stdout.flush()
   list_inc = list_rand*1 #convert to integer as they are indices, the set includes gene
   gene_c = gene_c+1
   
   
   if gene not in list_inc:
      list_inc.append(gene)
      
   list_exc=list_inc*1 
   list_exc.remove(gene)#indices that excludes gene
  
   
   x_train=train_data[:,list_inc]
   x_valid=valid_data[:,list_inc]
   x_test=test_data[:,list_inc]
   clf.fit(x_train,y_train)
    
   dec_val_test=[]

   dec_val_test=clf.decision_function(x_test)   
   auc_inc=roc_auc_score(y_test,dec_val_test)

   pred_val_test=clf.predict(x_test) 
   acc_inc=accuracy_score(y_test,pred_val_test)
   
   x_train=train_data[:,list_exc]
   x_valid=valid_data[:,list_exc]
   x_test=test_data[:,list_exc]
   clf.fit(x_train,y_train)
   #dlabel=clf.predict(x_test) 
    
   dec_val_test=[]
   dec_val_valid=[]

   dec_val_test=clf.decision_function(x_test)
   dec_val_valid=clf.decision_function(x_valid)
   
   auc_exc=roc_auc_score(y_test,dec_val_test)
   min_en1=entropy.entropy_of_vector(dec_val_valid,y_valid)
   
   pred_val_test=clf.predict(x_test) 
   acc_exc=accuracy_score(y_test,pred_val_test)
   
   
   pca_data = numpy.transpose(numpy.vstack((valid_data[:,gene],numpy.array(dec_val_valid))))
   pca_vector=pca2.PCA2(pca_data,1)
   min_en2=entropy.entropy_of_vector(pca_vector,y_valid)
   #print min_en1, min_en2
   ch=(auc_inc-auc_exc)*1.0
   chen=(min_en2-min_en1)*1.0
   #print ch,chen
   ch_acc=(acc_inc-acc_exc)*1.0
      
   ent.append([gene,chen])
   
   if ch>0.0 and chen>0.0:c[0]+=1
   elif ch<0.0 and chen>0.0:c[1]+=1
   elif ch<0.0 and chen<0.0:c[2]+=1
   elif ch>0.0 and chen<0.0:c[3]+=1
   
   tempch=(ch/(1.0 - auc_exc))*100.0
   tempchacc=(ch_acc/(1.0-acc_exc))*100
   del_auc.append(tempch)
   
   
   if ch !=0.0 and chen!=0.0:
   #   print gene, ch
      
      change.append(tempch)
      changeacc.append(tempchacc)
      ig.append(chen)
      ent_auc.append([gene,chen,tempch])
      if tempch<=0.0:
         auc_bool.append(1)
      else:
         auc_bool.append(0)

print
print "Base:",basen," Excluded:",gene_c," Total:",n
print c[0],c[3]
print c[1],c[2]


oddsratio, pvalue = stats.fisher_exact([[c[0], c[3]], [c[1], c[2]]])
print "Fishers Exact Test p-value: ",pvalue

filename_con=path+dataset+'.contable'
filename_rank=path+dataset+'.entrank'
filename_ent_auc=path+dataset+'.entauc'
numpy.savetxt(filename_con,c,delimiter=',')
title='Base AUC=%0.2f' % auc_base
numpy.savetxt(filename_rank, ent, delimiter=',')
numpy.savetxt(filename_ent_auc, ent_auc, delimiter=',')



#k=20
ent=numpy.array(ent)
feat=ent[numpy.argsort(ent[:,1])]
numpy.savetxt('ent', feat, delimiter=',')
#print wil_feat



#dx = [0.05,0.2,0.1]
dx=5
#plt.subplot(2,2,1)
plt.axhline(0,linewidth=0.5,ls='--',color="k")
plt.axvline(0,linewidth=0.5,ls='--',color="k")
plt.scatter(ig,change,s=dx,lw = 0.0,facecolor='0.0')
plt.ylim(ymax=100)
plt.xlim(plt.xlim()[::-1])
plt.title(title,fontsize=18)
plt.xlabel("Change in minimum entropy",fontsize=16)
plt.ylabel("\% Improvement in AUC",fontsize=16)
plt.savefig(figpath+'1.eps', format='eps', dpi=1200)
#plt.scatter(ig,change,s=dx,lw = 0.0,facecolor='0.65')
#plt.scatter(ig_noise,change_noise,s=10,color='k',marker="x",facecolor='0.10')
#plt.savefig(figpath+'5.eps', format='eps', dpi=1200)
plt.show()

