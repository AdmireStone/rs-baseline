# coding:utf8
import numpy as np
import random
import scipy.sparse as sp
import time

def predict(x,w0,W,V,sampler):
    '''
    :param x: (d,1) NOTE: the shape should not be (d,)
    :param w0:
    :param W:
    :param V:
    :return:
    如果x里面只有两个维度不是0，那么二次项计算，实际上只是这两个维度的交互。
    1. 获取x中非零元素下标
    2. 获取
    '''
    #ndarray [a,b]
    indx_feature=np.where(x==1)[0]
    num_features=len(indx_feature)

    qudratic_term=0.0
    for i in xrange(0,num_features-1):
        for j in xrange(i+1,num_features):
            v_feacture_1 = V[indx_feature[i]]
            v_feacture_2 = V[indx_feature[j]]
            qudratic_term += np.dot(v_feacture_1,v_feacture_2.T)


    linear_term = np.sum(W[indx_feature])

    y= w0 + linear_term + qudratic_term

    ## 回归的时候，确保不超出范围
    y = max(sampler.min_target,y)
    y = min(sampler.max_target, y)

    return y

def update_factors(w0,W,V,reg_w,reg_v,eta,x,y,sampler):
    '''
    目前只实现regression,采用 least square loss
    :param w0:the bias
    :param W:ndarray (D,1),where D is the dimensions of input vector
    :param V: ndarray (D,f), where f is the number of factors of latent vector
    :param reg_w:
    :param reg_v:
    :param eta:
    :param x : (d,1) NOTE: the shape should not be (d,)
    :param y : real labels
    :return:
    '''
    assert x.shape[1]==1


    y_p=predict(x,w0,W,V,sampler)

    error=2*(y_p-y)

    # update bais:
    w0-=eta*(1*error+2*w0*reg_w)

    # update w
    W-=eta*(x*error+2*W*reg_w)


    temp=sp.csr_matrix(W)

    # update V
    # V的更新实际上也涉及到一些矩阵相乘，但是其实，每次只有不为零的那些维度对应的V需要更新
    #
    indx_feature = np.where(x == 1)[0]
    num_features=len(indx_feature)
    num_factors=V.shape[1]
    v_deat=np.zeros((num_features,num_factors))
    for col in xrange(0,num_factors):
        v_deat[:col] = np.sum(V[:,col][indx_feature])
        v_deat[:col] -=V[:col]

    # deta=np.dot(np.dot(x,x.T),V)-V*(x**2)
    # assert deta.shape==V.shape
    # V-=eta*(deta*error+2*V*reg_v)

def init(num_factors,num_attribute,init_stdev,seed):
    W = np.zeros(num_attribute).reshape(-1,1)
    np.random.seed(seed=seed)
    V = np.random.normal(scale=init_stdev, size=(num_attribute, num_factors))
    return W,V

# todo
def loss():
    return 0.0

def train(sampler,hyper_args):
    """train model
    data: user-item matrix as a scipy sparse matrix
          users and items are zero-indexed
    """
    global W, V,w0

    reg_w=hyper_args['reg_w']
    reg_v=hyper_args['reg_v']
    eta=hyper_args['eta']
    num_factors=hyper_args['num_factors']
    w0=hyper_args['w0']
    num_iters=hyper_args['num_iters']
    init_stdev = hyper_args['init_stdev']
    seed=hyper_args['seed']

    num_users=sampler.num_users
    num_items=sampler.num_items

    # 这里将遗漏两个item
    total_dim=num_users+num_items



    W, V = init(num_factors,total_dim,init_stdev,seed)

    print 'initial loss = {0}'.format(loss())
    for it in xrange(num_iters):
        print 'starting iteration {0}'.format(it)
        sample_count=0
        for u,i,y in sampler.generate_samples():
            start=time.time()
            x=np.zeros((total_dim,))
            x[u]=1
            x[sampler.num_users+i]=1
            x=x.reshape(-1,1)
            up_start=time.time()
            update_factors(w0,W,V,reg_w,reg_v,eta,x,y,sampler)
            print "time:{0},function time={1}".format(time.time()-start,time.time()-up_start)
            sample_count+=1
            if sample_count % 5e3 ==0:
                print "Handle {0} of {1} samples".format(sample_count,sampler.num_samples)
        print 'iteration {0}: loss = {1}'.format(it,loss())
    return (w0,W,V)


def loadData(filename, path="/Users/dong/Desktop/BoostingFM-IJCAI18/dataset/ml-100k/"):
    data = []
    y = []
    users = set()
    items = set()
    with open(path + filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({"user_id": int(user), "movie_id": int(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (np.array(data), np.array(y), users, items)


class Data_Generator(object):

    def __init__(self,train_data,train_y,users,items,max_samples=None,isShuffle=True):
        data_size = len(train_y)
        if max_samples is None:
            self.num_samples = data_size
        else:
            self.num_samples = min(data_size, max_samples)

        self.train_data=train_data
        self.train_y=train_y
        self.num_users=len(users)
        self.num_items=len(items)
        # 回归的时候，预测的值不能超出范围
        self.min_target = min(train_y)
        self.max_target = max(train_y)
        # 打乱数据
        if isShuffle:
            idxs = range(self.num_samples)
            random.shuffle(idxs)
            self.train_data = train_data[idxs]
            self.train_y = train_y[idxs]

    def generate_samples(self):
        idx=0
        for _ in xrange(self.num_samples):
            u = self.train_data[idx]['user_id']
            i = self.train_data[idx]['movie_id']
            y = self.train_y [idx]
            idx += 1
            yield u, i, y


if __name__=="__main__":

    train_file="ua.base"
    train_data,train_y,users,items= loadData(train_file)

    sampler = Data_Generator(train_data,train_y,users,items)

    hyper_args={
    "reg_w":0.0025,
    "reg_v":0.0025,
    "eta":0.01,
    "num_factors":50,
    "w0":0.0,
    "num_iters":10,
    "init_stdev":0.1,
    "seed":28
    }

    train(sampler, hyper_args)

