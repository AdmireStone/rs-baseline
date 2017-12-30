# coding:utf8
import sys
from scipy.io import mmread
import numpy as np
from math import exp
import random
import pandas as pd
import sys
import pickle
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


def predict_b2b(W, Z, u, items):
    #     print type(items)
    items_onehot = sp.identity(1522, dtype='f', format='csr')[list(items)]
    # items_onehot = sp.identity(1682, dtype='f', format='csr')[list(items)]

    items_num = len(items)

    u_max = np.array([0.] * 943 * items_num).reshape(items_num, 943)
    u_max[:, u] = 1.
    u_max = sp.csr_matrix(u_max)

    eval_data = sp.csr_matrix(sp.hstack([u_max, items_onehot]))

    # np.dot(eval_data,W)
    linear_term = safe_sparse_dot(eval_data, W).T

    qusdratic_term = (safe_sparse_dot(safe_sparse_dot(eval_data, Z), eval_data.T)).diagonal()

    return np.asarray(linear_term + qusdratic_term)[0]


def getSingleContextAUC(s_c_ob, s_c_non_ob):
    acc = 0.0
    for s_ci in s_c_ob:
        acc += np.sum(s_c_non_ob < s_ci)
    return acc / (len(s_c_ob) * len(s_c_non_ob))


def getSingleMAPScore(ground_items, eval_items_list, score_list, k_arr):
    '''
    eval_items_list 和score_list 应该一一对应

    k 是计算前k个的map
    '''

    # 按照降序排序
    sorted_index = np.argsort(-score_list)
    eval_items_list = np.array(eval_items_list)
    sorted_items_list = eval_items_list[sorted_index]

    rst={}

    if k_arr==None:
        k=len(ground_items)
        num_precision = len(set(set(sorted_items_list[:k]) & set(ground_items))) * 1.
        rst['all']=num_precision / k
        return rst

    for k in k_arr:
        true_k = k
        if k > len(ground_items):
            k = len(ground_items)
        num_precision = len(set(set(sorted_items_list[:k]) & set(ground_items))) * 1.
        rst[str(true_k)] = num_precision / k
    return rst


def load_test_required_data():
    # test_datapath = "/home/zju/dgl/dataset/recommend/ml-100k/bprmf_test_0_1.txt"
    # tr_datapath = "/home/zju/dgl/dataset/recommend/ml-100k/bprmf_train_0_1.txt"

    test_datapath = "/Users/dong/Desktop/BoostingFM-IJCAI18/dataset/ml-100k/bprmf_test_0_1.txt"
    tr_datapath = "/Users/dong/Desktop/BoostingFM-IJCAI18/dataset/ml-100k/bprmf_train_0_1.txt"

    data = mmread(test_datapath).tocsr()
    train_data = mmread(tr_datapath).tocsr()

    users_index, items_index = data.nonzero()
    users_unique = np.unique(users_index)
    # 获取所有的items集合
    test_items_set = set(np.unique(items_index))
    _, ite = train_data.nonzero()
    train_items_set = set(np.unique(ite))
    items_set = train_items_set | test_items_set

    return users_unique, data, train_data, items_set


def predic_auc(W, Z):
    users_unique, data, train_data, items_set = load_test_required_data()
    totoal_auc = 0.0
    count = 0
    context_num = len(users_unique)
    # file_name="/home/zju/dgl/dataset/recommend/ml-100k/12_20_22_44/u1_iter_59_save_weight.pkl"
    # W,Z=loadB2bModel(weight_file)
    for u in users_unique:
        positive = set(data[u].indices)
        # 获取当前用户在训练数据集中的正标签
        train_pos = set(train_data[u].indices)
        neg = items_set - train_pos - positive
        s_c_ob = predict_b2b(W, Z, u, positive)
        s_c_non_ob = predict_b2b(W, Z, u, neg)
        acc = getSingleContextAUC(s_c_ob, s_c_non_ob)
        totoal_auc += acc
        count += 1
        sys.stdout.write("\r handl %d of %d" % (count, context_num))
        sys.stdout.flush()
    return totoal_auc / context_num


# def predic_map(W, Z, k):
#     users_unique, data, train_data, items_set = load_test_required_data()
#     total_map = 0.0
#     context_num = len(users_unique)
#     # file_name="/home/zju/dgl/dataset/recommend/ml-100k/12_20_22_44/u1_iter_59_save_weight.pkl"
#     # W,Z=loadB2bModel(weight_file)
#     for u in users_unique:
#         ground_items = set(data[u].indices)
#         if k > len(ground_items):
#             print "ground_items size={0},user={1}".format(len(ground_items), u)
#
#         # 获取当前用户在训练数据集中的正标签
#         train_pos = set(train_data[u].indices)
#
#         # items 要有序
#         eval_items_list = list(items_set - train_pos)
#         eval_items_list.sort()
#
#         score_list = predict_b2b(W, Z, u, eval_items_list)
#         user_map = getSingleMAPScore(ground_items, eval_items_list, score_list, k)
#
#         total_map += user_map
#
#     return total_map / context_num




def predict_bpr_score(user_factors,item_factors,u,items):
    items_list = list(items)
    user_vec = np.mat(user_factors[u])  # (1,N)
    items_matrix = np.mat(item_factors[items_list])  # (N,10)
    score_list=np.asarray(np.dot(items_matrix, user_vec.T)).T
    return score_list[0]

def predic_map_bpr(user_factors, item_factors, k_arr):
    users_unique, data, train_data, items_set = load_test_required_data()
    total_map = 0.0
    context_num = len(users_unique)
    # file_name="/home/zju/dgl/dataset/recommend/ml-100k/12_20_22_44/u1_iter_59_save_weight.pkl"
    # W,Z=loadB2bModel(weight_file)

    dict_MAP={}

    if k_arr !=None:
        for k in k_arr:
            dict_MAP[str(k)]=0.0
    else:
        dict_MAP['all'] = 0.0

    for u in users_unique:
        ground_items = set(data[u].indices)
        # if k > len(ground_items):
        #     print "ground_items size={0},user={1}".format(len(ground_items), u)

        # 获取当前用户在训练数据集中的正标签
        train_pos = set(train_data[u].indices)

        # items 要有序
        eval_items_list = list(items_set - train_pos)
        eval_items_list.sort()

        score_list = predict_bpr_score(user_factors,item_factors,u,eval_items_list)

        user_maps = getSingleMAPScore(ground_items, eval_items_list, score_list, k_arr)

        # try:
        for key in user_maps.keys():
            dict_MAP[key]+=user_maps[key]
        # except KeyError:
        #     print "KeyError,u={0},key={1}".format(u,key)
            # print

    for k in k_arr:
        dict_MAP[str(k)]/=context_num

    return dict_MAP

def predic_map_FM_like(W, Z, k_arr):
    users_unique, data, train_data, items_set = load_test_required_data()
    total_map = 0.0
    context_num = len(users_unique)
    # file_name="/home/zju/dgl/dataset/recommend/ml-100k/12_20_22_44/u1_iter_59_save_weight.pkl"
    # W,Z=loadB2bModel(weight_file)

    dict_MAP={}
    if k_arr !=None:
        for k in k_arr:
            dict_MAP[str(k)]=0.0
    else:
        dict_MAP['all'] = 0.0

    for u in users_unique:

        ground_items = set(data[u].indices)
        # if k > len(ground_items):
        #     print "ground_items size={0},user={1}".format(len(ground_items), u)

        # 获取当前用户在训练数据集中的正标签
        train_pos = set(train_data[u].indices)

        # items 要有序
        eval_items_list = list(items_set - train_pos)
        eval_items_list.sort()

        score_list = predict_b2b(W, Z, u, eval_items_list)

        user_maps = getSingleMAPScore(ground_items, eval_items_list, score_list, k_arr)

        # try:
        for key in user_maps.keys():
            dict_MAP[key]+=user_maps[key]
        # except KeyError:
        #     print "KeyError,u={0},key={1}".format(u,key)
            # print

    if k_arr !=None:
        for k in k_arr:
            dict_MAP[str(k)] /= context_num
    else:
        dict_MAP['all'] /=  context_num


    return dict_MAP


def mean_avg_precision(score_list,ground_truth_items,k_arr,exclued_items=None):
    '''
    :param score_list: a dict, holding the score of each item for the a certain context(user),{"str(item_id)":score} ...
                      e.g {"1":12.3,"2":16.3}
    :param ground_truth_items: the ground truth item list, all the items in which should be recommend
    :param exclued_items:  the items, which one does not want to be considered. like the items in the train data
    :param k_arr: an array or list,denoting how many items one want to evaluate e.g. [1,3,5]
    :return map_arr: a dict，each element is the result of MAP@k e.g {"1":0.3,"3":0.5,"5":0.6}

    1. 如果exclued_items不是空，则需要从sore_list删掉exclued_items对应的元素
    2. 对sore_list 按照值的大小排序，降序
    3. 统计前K元素的命中率
    '''


    if exclued_items !=None:
        for item_id in exclued_items:
            score_list.pop(str(item_id))

    # list e.g [('1',12.),('2',12)]
    score_list_ordered = sorted(score_list.iteritems(), key=lambda d:d[1], reverse=True)

    items_rank_list = [ int(user_score[0]) for user_score in score_list_ordered]

    map_arr = {}

    if k_arr==None:
        k=len(ground_truth_items)
        num_precision = len(set(items_rank_list[:k]) & set(ground_truth_items))
        map_arr['all']=(num_precision*1.) / k

    else:
        for k in k_arr:
            true_k = k
            if k > len(ground_truth_items):
                k = len(ground_truth_items)
            # 求前items_rank_list中前k个元素，与 ground_truth_items的交集
            num_precision = len(set(items_rank_list[:k]) & set(ground_truth_items))
            map_arr[str(true_k)] = (num_precision*1.) / true_k

    return map_arr


# def pre_rec(score_list,ground_truth_items,k_arr,exclued_items=None):
#     '''
#     计算准确率和查询率
#     :param score_list: a dict, holding the score of each item for the a certain context(user),{"str(item_id)":score} ...
#                       e.g {"1":12.3,"2":16.3}
#     :param ground_truth_items: the ground truth item list, all the items in which should be recommend
#     :param exclued_items:  the items, which one does not want to be considered. like the items in the train data
#     :param k_arr: an array or list,denoting how many items one want to evaluate e.g. [1,3,5]
#     :return (pre,rec): two dicts，respond to precision and recall  e.g pre={"1":0.3,"3":0.5,"5":0.6}
#
#     1. 如果exclued_items不是空，则需要从sore_list删掉exclued_items对应的元素
#     2. 对sore_list 按照值的大小排序，降序
#     3. 统计前K元素的命中率
#     '''
#     if exclued_items !=None:
#         for item_id in exclued_items:
#             score_list.pop(str(item_id))
#
#     # list e.g [('1',12.),('2',12)]
#     score_list_ordered = sorted(score_list.iteritems(), key=lambda d:d[1], reverse=True)
#
#     items_rank_list = [ int(user_score[0]) for user_score in score_list_ordered]
#
#     pre_arr={}
#     rec_arr = {}
#
#     for k in k_arr:
#         true_k = k
#         if k > len(ground_truth_items):
#             k = len(ground_truth_items)
#
#         # 求前items_rank_list中前k个元素，与 ground_truth_items的交集
#         num_precision = len(set(items_rank_list[:k]) & set(ground_truth_items))
#         map_arr[str(true_k)] = (num_precision*1.) / true_k
#
#     return map_arr



def test_mean_avg_precision():

    score_list={'0':12.2,'1':9.7,'3':10,'4':0.6,'5':0.2,'6':0.8,'7':120,}
    ground_truth_items=[1,4,3]
    exclued_items=None
    k_arr=[1,2,3]
    print  mean_avg_precision(score_list, ground_truth_items, None, exclued_items)



if __name__=="__main__":

    test_mean_avg_precision()