import numpy as np
import os

def save_data(img,label,modeldir='.'):
    img=np.array(img)
    img=img.reshape(1,-1)
    modeldir=os.path.join(modeldir,"num{}.txt".format(label))
    if os.path.exists(modeldir):
        model=np.loadtxt(modeldir)
        if len(model.shape)==1:
            model=model.reshape(1,-1)
        if model.shape[1]!=img.shape[1]:
            model=img
        else:
            model=np.append(model,img,axis=0)
    else:
        model=img
    print(model.shape)
    np.savetxt(modeldir, model, fmt='%d')

def train(modeldir='.'):
    data = []
    i=0
    count=0
    for j in range(10):
        numdir = os.path.join(modeldir, "num{}.txt".format(j))
        if os.path.exists(numdir):
            num = np.loadtxt(numdir)
        data.append(num)
    data=np.array(data)
    print(data.shape)
    w=np.zeros((data.shape[0],data.shape[2]+1))
    while True:
        if i==data.shape[0]*data.shape[1]:
            i=0
        if count==(data.shape[0]*data.shape[1]*10):
            break
        a=i%data.shape[0]
        b=i//data.shape[0]
        i+=1
        count+=1
        x=np.append(data[a][b],1)
        w0=w[a]
        index=[]
        for j in range(10):
            if j!=a and sum(w[j]*x)<sum(w0*x):
                continue
            if j==a:
                continue
            index.append(j)
        if len(index)!=0:
            w[a]+=x
        for j in index:
            w[j]-=x
    np.savetxt(os.path.join(modeldir, 'w.txt'), w, fmt='%d')

def test(img,modeldir='.'):
    ans=np.zeros(10)
    img =np.append(np.array(img),1)
    img = img.reshape(-1, )
    w_dir=os.path.join(modeldir,'w.txt')
    if os.path.exists(w_dir):
        w=np.loadtxt(w_dir)
        for i in range(10):
            ans[i]+=sum(w[i]*img)
    return np.argmax(ans)

#train()

