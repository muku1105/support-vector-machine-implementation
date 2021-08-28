import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv('diabetes_data_upload.csv')
xe=pd.get_dummies(df,columns=['Gender','Polyuria','Polydipsia','weakness','sudden weight loss','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity'])
xe.loc[xe['class'] == 'Positive', 'class'] = 1
xe.loc[xe['class'] == 'Negative', 'class'] = -1
xe['class']=xe['class'].astype(int)
x=xe.drop('class',axis=1).copy()
y=xe['class'].copy()
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=42,test_size=0.3)
print(x_train)
print(y_test)
x_train=np.mat(x_train)
y_train=np.mat(y_train)
x_test=np.mat(x_test)
y_test=np.mat(y_test)




max_passes=50 # initialize max_passes
c=1.0
tol=0.001 # initializing tol  
Minalpha=0.00001 
xarray=[]
yarray=[]
class support:
    x=[]
    y=[]
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self._alpha=np.mat(np.zeros((np.shape(x)[0],1)))
        passes=0 # initializing passes=0
        self._b=np.mat([0])
        self.smo()
        self.count=0
        self.mcount=0
        while(passes<max_passes):
            if(self.smo()==0):
                passes=passes+1
            else:
                passes=0

        
        
        


        
    def smo(self):
       num_changed_alphas=0
       for i in range(np.shape(self.x)[0]):
        
           ei=0
           ej=0
           for k in range(np.shape(self.x)[0]):
               # calculating Ei from equation 2.
               ei=ei+np.multiply(self.y[k],self._alpha[k]).T*(self.x[k]*self.x[i].T)
           ei=ei+self._b-self.y[i] 
                
               
           # calculating Ei from equation 2.
           
           
           # condition to check for KKT condition
           if((self.y[i]*ei<-tol and self._alpha[i]<c)or(self.y[i]*ei>tol and self._alpha[i]>0)):
               j=self.selectsecondindex(i,np.shape(self.x)[0])
               # calculating Ej using equation 2
               for k in range(np.shape(self.x)[0]):
                   ej=ej+np.multiply(self.y[k],self._alpha[k]).T*(self.x[k]*self.x[j].T)
               ej=ej+self._b-self.y[j]
               alphaiold=self._alpha[i].copy()
               alphajold=self._alpha[j].copy()
               # calculate L and H by 10 or 11
               LH=self.LHF(self._alpha[i], self._alpha[j],self.y[i],self.y[j])
               #calculating ETA by using (14)
               ETA=2.0*self.x[i]*self.x[j].T-self.x[i]*self.x[j].T-self.x[j]*self.x[j].T
               if(LH[0]!=LH[1]):
                   if(ETA<0):
                       if(self.optimizepair(i,j,ei,ej,ETA,LH,alphaiold,alphajold)):
                           num_changed_alphas+=1
                       
                   
                              
       return num_changed_alphas
    # compute value of alphaj by using equation (12) and (15) and check if |alphajold - alphaj|>10^(-5)
    def optimizepair(self,i,j,ei,ej,ETA,LH,alphaiold,alphajold):
        flag=False
        self._alpha[j]-=self.y[j]*(ei-ej)/ETA
        #compute and clip alphaj using 12 and 15
        if(self._alpha[j]<LH[0]):
            self._alpha[j]=LH[0]
        if(self._alpha[j]>LH[1]):
            self._alpha[j]=LH[1]
        if(abs(self._alpha[j]-alphajold)>=Minalpha):
            self._alpha[i]+=self.y[j]*self.y[i]*(alphajold-self._alpha[j])
            self.optimizeb(ei,ej,alphaiold,alphajold,i,j) # calculate value of b
            flag=True
        return flag
    # function to calculate b by using 17, 18 and 19
    def optimizeb(self,Ei,Ej,alphaiold,alphajold,i,j):
        b1=self._b-Ei-self.y[i]*(self._alpha[i]-alphaiold)*self.x[i]*self.x[i].T\
            -self.y[j]*(self._alpha[j]-alphajold)*self.x[i]*self.x[j].T
        b2=self._b-Ej-self.y[i]*(self._alpha[i]-alphaiold)*self.x[i]*self.x[j].T\
            -self.y[j]*(self._alpha[j]-alphajold)*self.x[j]*self.x[j].T
        if(0<self._alpha[i]) and (c>self._alpha[i]):
            self._b=b1
        elif(0<self._alpha[j]) and (c>self._alpha[j]):
            self._b=b2
        else:
            self._b=(b1+b2)/2.0
    #select j!= i randomly
    def selectsecondindex(self, indexoffirst,numberofrows):
        indexofsecond= indexoffirst
        while(indexoffirst==indexofsecond):
            indexofsecond= int(np.random.uniform(0,numberofrows))
        return indexofsecond

     # function to calculate L and H by 10 or 11       
    def LHF (self, alphai, alphaj,yi,yj):
        LH=[2]
        if(yi==yj):
            LH.insert(0, max( 0, 1))
            LH.insert(1, min( c, alphaj + alphai))
        else:
            LH.insert(0, max( 0, 2))
            LH.insert(0, min( c, alphaj - alphai + c))
        return LH
    def classify(self,x,y):
        
        cl=0
        for i in range(0,np.shape(self.x)[0]):
            cl=cl+(np.multiply(self._alpha[i],self.y[i]).T*(self.x[i]*x.T))
        cl=cl+self._b
            
        if(cl>0):
            if(y==1):
                self.count=self.count+1
            else:
                self.mcount=self.mcount+1
                
        else:
            if(y==-1):
                self.count=self.count+1
            else:
                self.mcount=self.mcount+1
    def display(self):
        print(self.count)
        print(self.mcount)
                
        
            
def testing(x_test,y_test):
    for i in range (np.shape(x_test)[0]):
        svm.classify(x_test[i],y_test[i])
    svm.display()
        
        
        
    

svm=support(x_train,y_train.transpose())
testing(x_test,y_test.transpose())
