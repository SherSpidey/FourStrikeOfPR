import numpy as np
import os
import math
#默认匹配图片大小10x14

def binalize(img,th=0.5):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]>=th:
                img[i][j]=1
            else:
                img[i][j]=0
    return img

def data_save(img,label,modeldir='.'):
    img=np.array(img)
    img=img.reshape(1,-1)
    countdir=os.path.join(modeldir,'count.txt')
    modeldir=os.path.join(modeldir,"num{}.txt".format(label))
    count=np.zeros(10)
    if os.path.exists(countdir):
        count=np.loadtxt(countdir)
    if os.path.exists(modeldir):
        model=np.loadtxt(modeldir)
        if len(model.shape)==1:
            model=model.reshape(1,-1)
        if model.shape[1]!=img.shape[1]:
            model=img
            count = np.zeros(10)
        else:
            model=np.append(model,img,axis=0)
    else:
        model=img
    print(model.shape)
    np.savetxt(modeldir, model, fmt='%d')
    count[label]+=1
    np.savetxt(countdir,count,fmt='%d')

def test(img,modeldir='.'):
    P=[]
    img=np.array(img)
    img=img.reshape(-1,1)
    countdir = os.path.join(modeldir, 'count.txt')
    count=np.loadtxt(countdir)
    for i in range(10):
        numdir = os.path.join(modeldir, "num{}.txt".format(i))
        if os.path.exists(numdir):
            model=np.loadtxt(numdir)
            S=np.cov(model,rowvar=False)
            Sr=np.linalg.pinv(S)
            det=np.linalg.det(S)
            if abs(det)<1e-5:
                det=1e-5
            x=np.mean(model,axis=0).reshape(-1,1)
            Pe=-(1/2)*np.matmul((img-x).T,np.matmul(Sr,img-x))-math.log(2*math.pi)*count[i]/2-math.log(abs(det))/2
            print(-(1/2)*np.matmul((img-x).T,np.matmul(Sr,img-x)),-math.log(abs(det))/2)
            P.append(Pe*count[i]/sum(count))
    P=P/sum(P)
    print(P)
    return np.argmax(P)




def train(img,label,modeldir='./model.txt',numdir='./num.txt'):
    img=np.array(img)
    img=img.reshape(1,-1)
    model = np.zeros((10, img.shape[1]))
    num = np.zeros(10)
    if os.path.exists(modeldir):
        model=np.loadtxt(modeldir)
    if os.path.exists(numdir):
        num=np.loadtxt(numdir)
    if model.shape[1] != img.shape[1]:
        model=np.zeros((10, img.shape[1]))
        num = np.zeros(10)
    model[label]+=img[0]
    num[label] += 1
    np.savetxt(modeldir,model,fmt='%f')
    np.savetxt(numdir,num,fmt='%d')

def test_no_G(img,modeldir='./model.txt',numdir='./num.txt'):
    img=np.array(img)
    img=img.reshape(1,-1)
    model=np.loadtxt(modeldir)
    num=np.loadtxt(numdir)
    pw=num/sum(num)
    pc=np.zeros((model.shape[0],model.shape[1]))
    pcw = np.zeros(10)
    for i in range(10):
        pc[i]=(model[i]+1)/(num[i]+2) #拉普拉斯校正
    for i in range(10):
        for j in range(model.shape[1]):
            if img[0][j]==0:
                pcw[i] +=math.log(1-pc[i][j])   #防止精度丢失+
            else:
                pcw[i] +=math.log(pc[i][j])
        pcw[i] =math.exp(pcw[i]*pw[i])
    pxw=pcw/sum(pcw)
    #print(pxw)
    return np.argmax(pxw)

