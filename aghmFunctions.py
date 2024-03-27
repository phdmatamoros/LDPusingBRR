#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:32:45 2022

@author: aghm
"""
#from distfit import distfit
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.fftpack import fft, dct
from scipy.fftpack import fft, dct
from scipy.stats import entropy
from collections import Counter
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn import mixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from numpy import *
from itertools import product  
from sklearn.utils.extmath import cartesian
from math import log
import math
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *
from pure_ldp.core import generate_hash_funcs
from sklearn.preprocessing import PolynomialFeatures
from numpy import linalg as LA
import pandas as pd
import numpy as np
import random 
import secrets
import math
#from bloom_aghm import BloomFilter
from pylab import imshow
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import init
warnings.filterwarnings('ignore')
from numpy import linalg as LA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
import seaborn as sb
import gc
#FUNCTIONS

import scipy as sp
from numpy import inf
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge,LassoCV,RidgeCV
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import random




def copula_gaussian(n, correlation,kwayr):
    zl=[]
    for w in range(kwayr):
        zl.append(sp.stats.norm.rvs(loc=0, scale=1, size=n,))
    Z = np.matrix(zl)
    # Construct the correlation matrix and Cholesky Decomposition
    rho = correlation
    cholesky = np.linalg.cholesky(rho)
    # Apply Cholesky and extract X and Y
    Z_XY = cholesky * Z
    val=[]
    acdf=[]
    for w in range(kwayr):
        val.append(np.array(Z_XY[w,:]).flatten())
        acdf.append(sp.stats.norm.cdf(np.array(Z_XY[w,:]).flatten(),loc=0, scale=1))
        #acdf.append(sp.stats.norm.pdf(np.array(Z_XY[w,:]).flatten(),loc=0, scale=1))
    return val,acdf 



def perturb_aghm(bit,fbp):
     #secretsGen=secrets.SystemRandom()
     p_sample=random.uniform(0, 1)#secretsGen.randint(0,100000)/100000
     sample=bit
     if p_sample < fbp:############################################################################################CHANGEEEE
         sample=random.choice([0,1])
     return sample

 
def str2vec(aux):
	v = []
	for s in aux:
		if(s in ['0', '1']): v.append(int(s))
	return(v)

def vector_aghm(bitmap,N,fbp):
    bmap=[]
    for ele in bitmap:
        bmap.append((ele-(fbp*N/2))/(1-fbp))
    return(np.array(bmap))



def df2np(bit2):
	bit3 = []
	for i in range(bit2.shape[0]):
		v = []
		for j in range(bit2.shape[1]):
			#print(i,j)
			v +=  str2vec(bit2.iloc[i,j])
		bit3.append(np.array(v, dtype=int))
	bit4 = np.array(bit3, dtype=int)
	return(bit4)

def df2nperturb(bit2,fbp):
    bit3 = []
    for i in range(bit2.shape[0]):
        v=[]
        for j in range(bit2.shape[1]):
            v += str2vec(bit2.iloc[i,j])
            v=(list(map(perturb_aghm, v,fbp*np.ones(len(bit2)))))
        bit3.append(np.array(v, dtype=int))
    bit4 = np.array(bit3, dtype=int)
    return(bit4)


def df2size(bit2):
    bit3 = [1]
    for i in range(len(bit2.columns)):
        v= str2vec(bit2.iloc[0,i])
        bit3.append(len(v))
    return(bit3)




def LASSO_original(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
        dfmul.append(value_cat[poi])


    if kway==1:
        dfmulQ = pd.DataFrame(dfmul[0])
        dfmulQ.columns = [attributes[0]]
        dfmulQ =(list(product(dfmulQ[attributes[0]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0]]
        ant = ant.astype('int')
        

        
    if kway==2: 
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1])).T
        dfmulQ.columns = [attributes[0],attributes[1]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(np.array(v))
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1]]
        ant = ant.astype('int')

        
    if kway==3:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2]]
        ant = ant.astype('int')

    if kway==4:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        ant = ant.astype('int')
        
    if kway==5:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3],dfmul[4])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]],dfmulQ[attributes[4]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        ant = ant.astype('int')

    
    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022


    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))
    

    Y=[]
    Y = np.sum(bit4, axis = 0)

    
    Y = vector_aghm(Y,len(bit4),fbp)
    

    clf=[]
    clf = Lasso(alpha = 0.1)#original 0.1
    clf.fit(M.T, Y)
    p_lasso=[]
    coef=abs(clf.coef_)#/len(bit4)
    p_lasso=coef/(sum(coef))

    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()
    p_lasso.columns = ["Lasso"]
    return p_lasso, coef/(sum(coef))


def Br_original(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
        dfmul.append(value_cat[poi])


    if kway==1:
        dfmulQ = pd.DataFrame(dfmul[0])
        dfmulQ.columns = [attributes[0]]
        dfmulQ =(list(product(dfmulQ[attributes[0]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0]]
        ant = ant.astype('int')

        
    if kway==2: 
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1])).T
        dfmulQ.columns = [attributes[0],attributes[1]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(np.array(v))
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1]]
        ant = ant.astype('int')

        
    if kway==3:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2]]
        ant = ant.astype('int')

    if kway==4:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        ant = ant.astype('int')
        
    if kway==5:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3],dfmul[4])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]],dfmulQ[attributes[4]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        ant = ant.astype('int')

    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022
    # print(valor)

    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))



    Y=[]
    Y = np.sum(bit4, axis = 0)
    Y = vector_aghm(Y,len(bit4),fbp)
    

    clf=[]
    clf=BayesianRidge(compute_score=True, n_iter=300, fit_intercept=False, alpha_init=fbp,tol=1e10)
    clf.fit(M.T, Y)
    p_lasso=[]
    coef=abs(clf.coef_)#/len(bit4)
    p_lasso=coef/(sum(coef))
    
    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()
    p_lasso.columns = ["Br"]
    return p_lasso, coef/(sum(coef))



def method(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
        dfmul.append(value_cat[poi])


    if kway==1:
        dfmulQ = pd.DataFrame(dfmul[0])
        dfmulQ.columns = [attributes[0]]
        dfmulQ =(list(product(dfmulQ[attributes[0]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0]]
        ant = ant.astype('int')

        
    if kway==2: 
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1])).T
        dfmulQ.columns = [attributes[0],attributes[1]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(np.array(v))
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1]]
        ant = ant.astype('int')

        
    if kway==3:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2]]
        ant = ant.astype('int')

    if kway==4:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        ant = ant.astype('int')
        
    if kway==5:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3],dfmul[4])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]],dfmulQ[attributes[4]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        ant = ant.astype('int')

    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022
    # print(valor)

    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    sizeme=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        bi4=dict_perturb[ele]
        sizeme.append(len(bi4[1,:]))
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))

   
    import operator

    possible=[]
    Ndir={}
    auxi=np.sum(M,axis=0)
    auxi[auxi >= 1] = 1
    for aka in bit4:
        arr=[]
        akc=aka*auxi
        # print(aka)
        # print('====================')
        for akb in M:
            # print(np.where(aka == 1)[0])
            # print(np.where(akb == 1)[0])
            # print(sum(abs(aka-akb)))
            # print(aka)
            # print(akb)
            # print(aka==akb)

            jiu=np.sum(np.array(akc==akb))#aka
            #/len(aka)
            
            #sum(abs(aka-akb))
            arr.append(jiu)
        
        # print('==========')
        # 
        arr=np.array(arr)
        # print(len(aka))
        # print(arr)
        # 1/0
        
        t = np.max(arr)==arr
        possible.append([i for i, x in enumerate(t) if x])
        Ndir['{}'.format([i for i, x in enumerate(t) if x])]=[]
    # 1/0 

    for element,aka in zip(possible,bit4):
        Ndir[str(element)].append(aka)
    # print(Ndir)
    # return Ndir
    # 1/0 
   
    keysList = list(Ndir.keys())
    import re
    
    fc=np.zeros(len(M))
    # print(fc)
    



    for keys in keysList:
        # print(keys)
        Nm=[]
        t=0
        res =[int(s) for s in re.findall(r'-?\d+\.?\d*', keys)]
        if len(res) == 1:
            bit4= Ndir[keys]
            fc[int(res[0])]=fc[int(res[0])]+len(bit4)
            # print(fc)
            

        if len(res) > 1:
            for key in res:
                # print(key)
                if t==0:
                    Nm=M[key]
                    t=1
                else:
                    Nm=np.vstack((Nm,M[key]))
            # print(Nm)
            bit4=[]
            bit4= Ndir[keys]
            Y=[]
            auxi=np.sum(Nm,axis=0)
            auxi[auxi >= 1] = 1
            # print(auxi)
            Y = np.sum(bit4, axis = 0)*auxi

            Y = vector_aghm(Y,len(bit4),fbp)
            

            clf=[]
            clf=BayesianRidge(compute_score=True, n_iter=300, fit_intercept=False, alpha_init=fbp,tol=1e10)
        
            clf.fit(Nm.T, Y)
            p_lasso=[]
            coef=abs(clf.coef_)#/len(bit4)
            p_lasso=coef/(sum(coef))
            # print(res)
            # print(p_lasso*len(bit4))
            
            for xres,xlas in zip(res,p_lasso):
                fc[int(xres)]=fc[int(xres)]+float(xlas)*len(bit4)
            # print(fc)
        
        
            
    p_lasso=fc/sum(fc) 
    # print(p_lasso)
    # print(index)
    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()  
    p_lasso.columns = ["other"]


   
    
    return p_lasso


 

def RF(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
        dfmul.append(value_cat[poi])


    if kway==1:
        dfmulQ = pd.DataFrame(dfmul[0])
        dfmulQ.columns = [attributes[0]]
        dfmulQ =(list(product(dfmulQ[attributes[0]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0]]
        ant = ant.astype('int')

        
    if kway==2: 
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1])).T
        dfmulQ.columns = [attributes[0],attributes[1]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(np.array(v))
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1]]
        ant = ant.astype('int')

        
    if kway==3:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2]]
        ant = ant.astype('int')

    if kway==4:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        ant = ant.astype('int')
        
    if kway==5:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3],dfmul[4])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]],dfmulQ[attributes[4]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        ant = ant.astype('int')

    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022
    # print(valor)

    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))



    Y=[]
    Y = np.sum(bit4, axis = 0)


    

    clf=[]
    clf=RandomForestClassifier(n_estimators=100)

    clf.fit(M.T, Y)
    p_lasso=[]
    coef=abs(clf.feature_importances_)#/len(bit4)
    p_lasso=coef/(sum(coef))
    
    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()
    p_lasso.columns = ["Rf"]
    

    
    
    
    
    
    
    
    return p_lasso, coef/(sum(coef))


def methodrf(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
        dfmul.append(value_cat[poi])


    if kway==1:
        dfmulQ = pd.DataFrame(dfmul[0])
        dfmulQ.columns = [attributes[0]]
        dfmulQ =(list(product(dfmulQ[attributes[0]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0]]
        ant = ant.astype('int')

        
    if kway==2: 
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1])).T
        dfmulQ.columns = [attributes[0],attributes[1]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(np.array(v))
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1]]
        ant = ant.astype('int')

        
    if kway==3:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2]]
        ant = ant.astype('int')

    if kway==4:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
        ant = ant.astype('int')
        
    if kway==5:
        dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3],dfmul[4])).T
        dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]],dfmulQ[attributes[4]])))
        lista=[]
        for v in dfmulQ:
            if sum(np.isnan(v)) <= 0:
                lista.append(v)
        ii=np.array(lista)
        tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        ant = ant.astype('int')

    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022
    # print(valor)

    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    sizeme=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        bi4=dict_perturb[ele]
        sizeme.append(len(bi4[1,:]))
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))

   
    import operator

    possible=[]
    Ndir={}
    auxi=np.sum(M,axis=0)
    auxi[auxi >= 1] = 1
    for aka in bit4:
        arr=[]
        akc=aka*auxi
        # print(aka)
        # print('====================')
        for akb in M:
            # print(np.where(aka == 1)[0])
            # print(np.where(akb == 1)[0])
            # print(sum(abs(aka-akb)))
            # print(aka)
            # print(akb)
            # print(aka==akb)

            jiu=np.sum(np.array(akc==akb))#aka
            #/len(aka)
            
            #sum(abs(aka-akb))
            arr.append(jiu)
        
        # print('==========')
        # 
        arr=np.array(arr)
        # print(len(aka))
        # print(arr)
        # 1/0
        
        t = np.max(arr)==arr
        possible.append([i for i, x in enumerate(t) if x])
        Ndir['{}'.format([i for i, x in enumerate(t) if x])]=[]
    # 1/0 

    for element,aka in zip(possible,bit4):
        Ndir[str(element)].append(aka)
    # print(Ndir)
    # return Ndir
    # 1/0 
   
    keysList = list(Ndir.keys())
    import re
    
    fc=np.zeros(len(M))
    # print(fc)
    



    for keys in keysList:
        # print(keys)
        Nm=[]
        t=0
        res =[int(s) for s in re.findall(r'-?\d+\.?\d*', keys)]
        if len(res) == 1:
            bit4= Ndir[keys]
            fc[int(res[0])]=fc[int(res[0])]+len(bit4)
            # print(fc)
            

        if len(res) > 1:
            for key in res:
                # print(key)
                if t==0:
                    Nm=M[key]
                    t=1
                else:
                    Nm=np.vstack((Nm,M[key]))
            # print(Nm)
            bit4=[]
            bit4= Ndir[keys]
            Y=[]
            auxi=np.sum(Nm,axis=0)
            auxi[auxi >= 1] = 1
            # print(auxi)
            Y = np.sum(bit4, axis = 0)*auxi

            # Y = vector_aghm(Y,len(bit4),fbp)
            

            clf=[]
            clf=clf=RandomForestClassifier(n_estimators=100)#BayesianRidge(compute_score=True, n_iter=300, fit_intercept=False, alpha_init=fbp,tol=1e10)
        
            clf.fit(Nm.T, Y)
            p_lasso=[]
            coef=abs(clf.feature_importances_)#/len(bit4)
            p_lasso=coef/(sum(coef))
            # print(res)
            # print(p_lasso*len(bit4))
            
            for xres,xlas in zip(res,p_lasso):
                fc[int(xres)]=fc[int(xres)]+float(xlas)*len(bit4)
            # print(fc)
        
        
            
    p_lasso=fc/sum(fc) 
    # print(p_lasso)
    # print(index)
    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()  
    p_lasso.columns = ["other_rf"]


   
    
    return p_lasso
