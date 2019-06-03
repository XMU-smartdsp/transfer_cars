# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:05:58 2019

@author: Design
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
import layer as l
import unit 
import numpy as np
import tensorflow as tf
from random import shuffle
from copy import deepcopy as dc
MODEL_INIT = np.load('/home/happywjt/Active_Learning/Alexnet/bvlc_alexnet_new.npy').item()     #初始化载入的模型
BATCH_SIZE = 64                                                                                #batchsize
CLASS_NUM=20                                                                                   #类目数
EPOCH = 100                                                                                    #训练迭代次数
source_dir = ''                                          #源域路径
target_dir = ''                                            #目标域路径
item_id = [1,15,20,35,48,60,73,86,111,124,137,162,175,188,212,225,238,250,263,276]
item_label = [0,4,8,15,16,17,18,19,1,2,3,5,6,7,9,10,11,12,13,14]
MODEL_NAME = 'model/od_svtransmodel.ckpt'

#网络模型的设计部分，分类网络在此设置。
def SharePart(input, drop_out_rate, reuse = False):
    
    def pre_process(input):
        rgb_scaled = input
        Mean = [103.939,116.779,123.68]
        
        red,green,blue = tf.split(rgb_scaled,3,3)
        bgr = tf.concat([
                red - Mean[2],
                green - Mean[1],
                blue - Mean[0]],3)
        return bgr
    
    input = pre_process(input)
    
    with tf.variable_scope('Share_Part',reuse = reuse):
        
        conv1 = l.conv2d('conv1',input,(11,11),96,strides = [1,4,4,1],decay = (0.0,0.0),pad='VALID',Init = MODEL_INIT['conv1'])
        maxpool1 = l.max_pooling('maxpool',conv1,3,2)
        norm1 = tf.nn.lrn(maxpool1,depth_radius=2,alpha=2e-05,beta=0.75,name='conv1')
    
        conv2 = l.conv2d_with_group('conv2',norm1,(5,5),256,2,decay = (0.0,0.0),pad = 'SAME', Init = MODEL_INIT['conv2'])
        maxpool2 = l.max_pooling('maxpool2',conv2,3,2)
        norm2 = tf.nn.lrn(maxpool2,depth_radius=2,alpha=2e-05,beta=0.75,name='conv2')

        conv3 = l.conv2d('conv3',norm2,(3,3),384,pad = 'SAME',Init = MODEL_INIT['conv3'])
    
    
        conv4 = l.conv2d_with_group('conv4',conv3,(3,3),384,2,pad = 'SAME',Init = MODEL_INIT['conv4'])
       
        conv5 = l.conv2d_with_group('conv5',conv4,(3,3),256,2,pad = 'SAME',Init = MODEL_INIT['conv5'])
        maxpool5 = l.max_pooling('maxpool5',conv5,3,2)
        print (maxpool5.shape)
    
        dim=1
        shape = maxpool5.get_shape().as_list()
        for d in shape[1:]:
            dim*=d
    
        reshape = tf.reshape(maxpool5,[-1,dim])
    
        fc6 = l.fully_connect('fc6',reshape,4096,Init = MODEL_INIT['fc6'])
        fc6 = l.dropout('drop_6',fc6,drop_out_rate)
        fc7 = l.fully_connect('fc7',fc6,4096,Init = MODEL_INIT['fc7'])
        fc7 = l.dropout('drop_7',fc7,drop_out_rate)
        
    return fc7

#网络的任务层也就是softmax层
def MissionPart(input, reuse=False):
    
    with tf.variable_scope('Classifier',reuse= reuse):
        result = l.fully_connect('classifier',input,CLASS_NUM,active=None)
    return result

#网络层计算损失函数部分
def SoftmaxWithLoss(logistic,label):
    
    label = tf.one_hot(label,depth = CLASS_NUM)
    loss = tf.losses.softmax_cross_entropy(label,logistic)
    
    return loss

#MMD层，参数包括源域数据的输入，目标域数据的输入，mmd系数，核值，核个数，是否为固定sigma
def MMDPart(source_input,target_input,mmd_alpha = 0.3,kernel_mul=2.0, kernel_num=5.0, fix_sigma=None, name='MMD'):
    if mmd_alpha == 0:
        return 0
    with tf.variable_scope(name):
        total = tf.concat([source_input,target_input],0)
        total_size = tf.shape(total)
        total0 = tf.reshape(tf.tile(total,[total_size[0],1]),[total_size[0],total_size[0],total_size[1]])
        total1 = tf.reshape(tf.tile(total,[1,total_size[0]]),[total_size[0],total_size[0],total_size[1]])
        L2_distance = tf.reduce_sum((total0-total1)**2,-1)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum((L2_distance)/(tf.to_float(total_size[0]*total_size[0]-total_size[0])))
        bandwidth /= kernel_mul ** (kernel_num//2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(int(kernel_num))]
        kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernels = tf.reduce_sum(kernel_val,0)
        XX = kernels[:BATCH_SIZE, :BATCH_SIZE]
        YY = kernels[BATCH_SIZE:,BATCH_SIZE:]
        XY = kernels[:BATCH_SIZE,BATCH_SIZE:]
        YX = kernels[BATCH_SIZE:,:BATCH_SIZE]
        loss = tf.multiply(mmd_alpha,tf.reduce_mean(XX+YY-XY-YX),name="loss")
        tf.add_to_collection('losses',loss)
        return loss
		
#训练网络时所调节的网络层参数以及优化方法
def train_net(loss, base_lr=0.0001):
    
    
    var_list = tf.trainable_variables()
    trn_list = []                                       
	#注意这里，因为载入了预训练模型，所以conv1和conv2并未加入trn_list，也就是不更新参数的，如果不载入预训练模型，记得修改这里，将conv1和conv2加入trnlist
    for i in var_list:
        if 'conv1' not in i.name and 'conv2' not in i.name:
            trn_list.append(i)
            tf.summary.histogram('weight',i)
    
    loss = tf.add_n(tf.get_collection('losses'),name='all_loss')
    opt = tf.train.AdamOptimizer(base_lr).minimize(loss,var_list=trn_list)
    return opt

#测试函数
def Test(logistic,label):
    
    result = tf.cast(tf.argmax(logistic,axis = 1),tf.uint8)
    compare = tf.cast(tf.equal(result,label),tf.float32)
    acc = tf.reduce_mean(compare)
    return acc
	
if __name__ == '__main__':
    source_samples_list = unit.get_list(item_id,item_label,source_dir)  
    target_samples_list = unit.get_list(item_id,item_label,target_dir)
    
    batch_s = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    batch_t = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label_s = tf.placeholder(tf.uint8,[None])
    label_t = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    feature_s = SharePart(batch_s,keep_prop)
    feature_t = SharePart(batch_t,keep_prop,reuse = True)
    result_s = MissionPart(feature_s)
    result_t = MissionPart(feature_t,reuse=True)
    loss_s = SoftmaxWithLoss(result_s,label_s)
#    loss_t = SoftmaxWithLoss(result_t,label_t)
    mmdloss1 = MMDPart(feature_s,feature_t,mmd_alpha=0.3,name='MMD1')                #feature_s为fc7输出，送入第一个mmd层计算
    mmdloss2 = MMDPart(result_s,result_t,mmd_alpha=0.01,name='MMD2')                 #result_s为fc8输出，送入第二个mmd层计算
    acc_train = Test(result_s,label_s)
    acc_test = Test(result_t,label_t)
    opt = train_net(loss_s)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    sess.run(init)
    print "begin"
    best_target_acc = 0
    for i in range(EPOCH):
        shuffle(source_samples_list)
        shuffle(target_samples_list)
        source_accuracy = 0
        target_accuracy = 0
        target_cost = 0
        
        for j in range(len(source_samples_list)/BATCH_SIZE):
            sourcebatch,sourcelabels = unit.GetBatch(source_samples_list,BATCH_SIZE,j*BATCH_SIZE)
            targetbatch,targetlabels = unit.GetBatch(target_samples_list,BATCH_SIZE,j*BATCH_SIZE)
            sess.run(opt,feed_dict={batch_s:sourcebatch,batch_t:targetbatch,label_s:sourcelabels,keep_prop:0.5})
            source_accuracy += sess.run(acc_train,feed_dict={batch_s:sourcebatch,label_s:sourcelabels,label_t:targetlabels,keep_prop:1.0})
#            print sess.run(loss_t,feed_dict={batch_t:targetbatch, label_t:targetlabels,keep_prop:1.0})
        for j in range(len(target_samples_list)/BATCH_SIZE):
            targetbatch,targetlabels = unit.GetBatch(target_samples_list,BATCH_SIZE,j*BATCH_SIZE)
            target_accuracy+=sess.run(acc_test ,feed_dict={batch_t:targetbatch, label_t:targetlabels,keep_prop:1.0})
#            target_cost+=sess.run(loss_t,feed_dict={batch_t:targetbatch, label_t:targetlabels,keep_prop:1.0})
            
        source_accuracy /= (len(source_samples_list)/BATCH_SIZE)
        target_accuracy /= (len(target_samples_list)/BATCH_SIZE)
#        target_cost /= (len(target_samples_list)/BATCH_SIZE)
        
        print "this is the ",i," epoch"
        print "target accuracy is: ",target_accuracy
        print "source accuracy is: ",source_accuracy
#        print "target cost is: ",target_cost
        
        if target_accuracy>best_target_acc:
            best_target_acc = target_accuracy
            saver.save(sess,MODEL_NAME)
            print "the best target acc is:",best_target_acc
        else:
            print "the best target acc is:",best_target_acc        