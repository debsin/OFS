import numpy
import math
import operator


def part_entropy(pclass,l):
    n=l
    c1=pclass.count(1)
    c0=n-c1
    if c0==0 or c1==0: # 0*log0 or 1*log1
       return 0.0
    else:
       p0=float(c0)/n
       p1=float(c1)/n
       return -(   (p0*math.log(p0,2))   +   (p1*math.log(p1,2))   )
    
def entropy_of_vector(vector,myclass):
    #print vector
    #print myclass
    mylist = []
    n=len(myclass)
    #cp1=myclass.count(1)
    #cp0=n-cp1
    #print n, cp1, cp0
    
    for i in range(0,len(vector)):
	mylist.append([myclass[i],vector[i]])
    mylist=sorted(mylist, key=operator.itemgetter(1)) #sort 2Dlist according to 2nd column
    #mylist=numpy.matrix(mylist)
    myclass=[]
    vector=[]
    
    myclass=[row[0] for row in mylist]
    #myclass=mylist[:,0]
    #vector=mylist[:,1]
    #print myclass
    remain=[]
    for i in range(n-1):#for each threshold
       #threshold=sum(vector[i:i+2],0)/2
       #print threshold[0,0]
       #partA=[]
       #partB=[]
       classA=[]
       classB=[]
       lenA=0.0
       lenB=0.0
       classA=myclass[0:i+1]
       classB=myclass[i+1:n]
       #print classA,classB
       lenA=(i+1.0) 		#len(classA)*1
       lenB=(n-(i+1.0)) 	#len(classB)*1
       """
       for p in range(n):
          if vector[p,0]<threshold[0,0]:
             #partA.append(vector[p,0])
             classA.append(myclass[p,0])
             lenA+=1.0
          else:
             #partB.append(vector[p,0])
	     classB.append(myclass[p,0])
	     lenB+=1.0
       #print partA
       """
       epA=part_entropy(classA,lenA)
       epB=part_entropy(classB,lenB)
       #print epA,epB
       remain_t= (lenA/n*epA) + (lenB/n*epB)
       #print lenA,lenB, remain_t
       remain.append(remain_t)
       #print "Min",min(remain)
       
    return min(remain)

myvals=[72.,59.,58.,22.,89.]
mylabels=[0,1,0,1,0]
#print entropy_of_vector(myvals,mylabels)
