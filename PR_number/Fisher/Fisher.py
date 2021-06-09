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

def train(modeldir='.'):            #采用OvR分类法
    data=[]
    W=[]                    #保存w
    std=[]                  #保存分类标准
    for i in range(10):
        numdir = os.path.join(modeldir, "num{}.txt".format(i))
        if os.path.exists(numdir):
            num=np.loadtxt(numdir)
        data.append(num)
    data=np.array(data)
    print(data.shape)
    for i in range(10):
        Sw0=np.zeros((data[0].shape[1],data[0].shape[1]))
        Sw1=np.zeros((data[0].shape[1],data[0].shape[1]))
        X0=data[i]
        X1=np.delete(data,i,axis=0).reshape(-1,X0.shape[1])
        u0=np.mean(X0,axis=0).reshape(-1,1)
        u1=np.mean(X1,axis=0).reshape(-1,1)
        for x in X0:
            x=x.reshape(-1,1)
            Sw0+=np.matmul(x-u0,(x-u0).T)
        for x in X1:
            x=x.reshape(-1,1)
            Sw1+=np.matmul(x-u1,(x-u1).T)
        Sw=Sw0+Sw1
        Swi=np.linalg.inv(Sw)
        w=np.matmul(Swi,(u0-u1))
        a=np.matmul(w.T,u0)
        b=np.matmul(w.T,u1)
        W.append(w.reshape(1,-1))
        std.append([a,b])
    W=np.array(W)
    std=np.array(std)
    np.savetxt(os.path.join(modeldir,'W.txt'),W.reshape(W.shape[0],W.shape[2]),fmt='%f')
    np.savetxt(os.path.join(modeldir, 'std.txt'), std.reshape(-1,2), fmt='%f')

def test(img,modeldir='.'):
    ans=[]
    img = np.array(img)
    img = img.reshape(-1, 1)
    W_dir=os.path.join(modeldir,'W.txt')
    std_dir=os.path.join(modeldir,'std.txt')
    if os.path.exists(W_dir):
        W=np.loadtxt(W_dir)
        if os.path.exists(std_dir):
            std=np.loadtxt(std_dir)
            for i in range(10):
                x=np.matmul(W[i].reshape(1,-1),img)
                """
                if abs(x-std[i][0])<abs(x-std[i][1]):
                    ans.append(abs(x-std[i][0])[0])
                else:
                    ans.append(99*np.ones(1))"""
                ans.append(abs(x-std[i][0])/abs(std[i][0]-std[i][1]))

    ans=np.array(ans).reshape(1,-1)
    return np.argmin(ans)



