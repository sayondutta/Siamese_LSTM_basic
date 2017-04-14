# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import random


# In[2]:

#loading data from quora2vec embeddings
#train_q1_sentence_array = np.load('/mnt/disks/tensorflow/train_q1_sentence_array_q2v.npy')
#train_q2_sentence_array = np.load('/mnt/disks/tensorflow/train_q2_sentence_array_q2v.npy')
#test_q1_sentence_array = np.load('/mnt/disks/tensorflow/test_q1_sentence_array_q2v.npy')
#test_q2_sentence_array = np.load('/mnt/disks/tensorflow/test_q2_sentence_array_q2v.npy')
#embedding_array = np.load('/mnt/disks/tensorflow/embedding_array.npy')

#print train_q1_sentence_array[1]
#print train_q2_sentence_array[1]
#print test_q1_sentence_array[1]
#print test_q2_sentence_array[1]
#print embedding_array[1]
# In[2]:




labels = np.load('/mnt/disks/tensorflow/labels.npy')
#print labels[:100]
#exit()
# In[3]:

#loading data from glove vectors embeddings
train_q1_sentence_array = np.load('/mnt/disks/tensorflow/train_q1_sentence_array_glove.npy')
train_q2_sentence_array = np.load('/mnt/disks/tensorflow/train_q2_sentence_array_glove.npy')
test_q1_sentence_array = np.load('/mnt/disks/tensorflow/test_q1_sentence_array_glove.npy')
test_q2_sentence_array = np.load('/mnt/disks/tensorflow/test_q2_sentence_array_glove.npy')
embedding_array = np.load('/mnt/disks/tensorflow/embedding_array_glove.npy')


# In[4]:

print train_q1_sentence_array.shape
print train_q2_sentence_array.shape
print test_q1_sentence_array.shape
print test_q2_sentence_array.shape
print embedding_array.shape
print labels.shape
#print train_q1_sentence_array_glove.shape
#print train_q2_sentence_array_glove.shape
#print test_q1_sentence_array_glove.shape
#print test_q2_sentence_array_glove.shape
#print embedding_array_glove.shape


# In[5]:

#random.seed(1)
valid_ids = random.sample(range(train_q1_sentence_array.shape[0]),50000)
train_ids = list(set(range(train_q1_sentence_array.shape[0]))-set(valid_ids))
#print valid_ids[:100]
#print train_ids[:100]

# In[6]:

#dependency functions
#creating batches

def create_batches(batch_size,data_set):
    random_ids = random.sample(range(data_set.shape[0]),batch_size)
    return random_ids

#accuracy calculator
def predictions(ypred_inp,t=0.7):
    #ypred = np.array([np.argmax(i) for i in ypred_inp])
    ypred = []
    for i in ypred_inp:
	if i<t:
		ypred.append(0.0)
	else:
		ypred.append(1.0)
    ypred = np.asarray(ypred)
    return ypred
    
def accuracy(ypred,yact,t=0.7):
    ypred = predictions(ypred,t)
    #yact = predictions(yact)
    #print np.sum(ypred),np.sum(yact),len(list(yact))
    return np.mean(np.equal(ypred,yact))*100

def arrange_outputs(labels):
    label_modified = []
    for i in labels:
        if i == 0:
            label_modified.append([1,0])
        else:
            label_modified.append([0,1])
    label_modified = np.asarray(label_modified)
    return label_modified

# actual length of the sentences without padding
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    print "sequence length tf shape:",length.shape
    return length


# In[7]:

labels_new = arrange_outputs(labels)
#print labels_new.shape
#exit()

# In[9]:

#hyperparameters
batch_size = 150
iterations = 600001
learning_rate = 0.01
num_steps = train_q1_sentence_array.shape[1]
num_features = embedding_array.shape[1]
num_classes = labels.shape[1]
hidden_units = num_features
activation_units = 2


# In[10]:

#setting up the computational graphs
 # data input tensors

x1 = tf.placeholder(dtype=tf.int64,shape=[None,num_steps])
x2 = tf.placeholder(dtype=tf.int64,shape=[None,num_steps])
y = tf.placeholder(dtype=tf.float32,shape=[None,num_classes])
    #post embeddings tensors
embedding_tensor = tf.constant(embedding_array,dtype=tf.float32)
x1_inputs = tf.nn.embedding_lookup(embedding_tensor,x1)
x2_inputs = tf.nn.embedding_lookup(embedding_tensor,x2)
    # weights tensors
w1 = {'in' : tf.Variable(tf.truncated_normal([num_features,hidden_units],stddev=0.1)),
         'out' : tf.Variable(tf.truncated_normal([hidden_units,activation_units],stddev=0.1)),
         'fc' : tf.Variable(tf.truncated_normal([activation_units,num_classes],stddev=0.1))
         }
b1 = {'in': tf.Variable(tf.ones([hidden_units,])),
         'out' : tf.Variable(tf.ones([activation_units,])),
         'fc' : tf.Variable(tf.constant(1.0,shape=[num_classes]))
         }

w2 = {'in' : tf.Variable(tf.random_normal([num_features,hidden_units])),
         'out' : tf.Variable(tf.random_normal([hidden_units,activation_units]))
         }
b2 = {'in': tf.Variable(tf.ones([hidden_units,])),
         'out' : tf.Variable(tf.ones([activation_units,]))
         }

    #x1_inputs = tf.reshape(x1_inputs,[-1,num_features])
    #x2_inputs = tf.reshape(x2_inputs,[-1,num_features])
#x1_in = x1_inputs#tf.matmul(x1_inputs,w1['in'])+b1['in']
    #x1_in = tf.reshape(x1_in,[-1,num_steps,hidden_units])
#x2_in = x2_inputs#tf.matmul(x2_inputs,w2['in'])+b2['in']
    #x2_in = tf.reshape(x2_in,[-1,num_steps,hidden_units])
    # lstm
lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_units)
    
    # dynamic rnn 1
with tf.variable_scope("dynamicrnns") as scope:
     o1,ls1 = tf.nn.dynamic_rnn(cell=lstm_cell1,dtype=tf.float32,inputs=x1_inputs,sequence_length=length(x1_inputs))
    # dynamic rnn 2
     scope.reuse_variables()
     o2,ls2 = tf.nn.dynamic_rnn(cell=lstm_cell1,dtype=tf.float32,inputs=x2_inputs,sequence_length=length(x2_inputs))

#o1 = tf.unstack(tf.transpose(o1,[1,0,2]))
    #r1 = o1[-1]#tf.matmul(o1[-1],w1['out'])+b1['out']
    #r1 = tf.nn.softmax(r1)
#o2 = tf.unstack(tf.transpose(o2,[1,0,2]))
    #r2 = o2[-1]#tf.matmul(o2[-1],w2['out'])+b2['out']
    #r2 = tf.nn.softmax(r2)
    # activation on the absolute difference of the output activation from both the rnns
#print ls2[0].shape,ls2[1].shape
#print ls1[0].shape,ls1[1].shape
#exit()
dist = tf.square(tf.subtract(ls1[1],ls2[1]))
dist = tf.reduce_sum(dist,axis=1)
#dist = dist+epsilon
#print "state lstm1:",ls1[1].shape
#print "state lstm2:",ls2[1].shape
#print dist_.shape
#print dist__.shape
    #diff = tf.square(diff)
#dist___ = tf.sigmoid(dist__)#tf.exp(tf.multiply(-1.0,dist__))
#print dist___.shape
#######
angle_num = tf.reduce_sum(tf.multiply(ls1[1],ls2[1]),axis=1)
#angle_num = angle_num + epsilon
#angle_den = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(ls1[1]),axis=1)),tf.sqrt(tf.reduce_sum(tf.square(ls2[1]),axis=1)))
#angle_den = angle_den + epsilon
#cos_theta = tf.div(angle_num,angle_den)
#cos_theta = cos_theta + epsilon
cos_theta = angle_num
cos_theta = tf.reshape(cos_theta,[-1,1])
dist = tf.reshape(dist,[-1,1])
output_mat = tf.concat([dist,cos_theta],axis=1)
#output_mat = output_mat + epsilon
#######
#print dist.shape
#exit()
#print output_mat.shape
#print angle_num.shape
#print angle_den.shape
#print cos_theta.shape
#exit()
#dist = tf.reshape(dist,[-1,1])
output = tf.matmul(output_mat,w1['fc'])+b1['fc'] #output
output = tf.exp(-1*output)
#epsilon = tf.constant(0.001,shape=[-1,1])
#output = output+epsilon
#output = tf.nn.softmax(output)
#print output.shape
#exit()
    # loss function : log loss
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))
loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(output-y),axis=1)))
#loss = loss + epsilon
#print loss.shape
#exit()
    #training
optimiser = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[ ]:

#initialising tensors and session run
random.shuffle(train_ids)
train_q1 = train_q1_sentence_array[train_ids]
#print train_q1.shape
train_q2 = train_q2_sentence_array[train_ids]
#print train_q2.shape
labels_train = labels[train_ids]
#print labels_train.shape
#exit()
valid_q1 = train_q1_sentence_array[valid_ids]
valid_q2 = train_q2_sentence_array[valid_ids]
labels_valid = labels[valid_ids]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    #init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step*batch_size < iterations:
        random_ids = create_batches(batch_size,train_q1)
	#print sum(random_ids)
        #print random_ids
	#step+=1
	#continue
	_,l,predis = sess.run([optimiser,loss,dist],feed_dict = {x1 : train_q1[random_ids],
                                        x2 : train_q2[random_ids],
                                        y : labels_train[random_ids]})
        #	print step
        if step % 100 == 0:
            #tr_pred = sess.run(pred,feed_dict = {x1 : train_q1,x2 : train_q2})
            #tr_acc = accuracy(tr_pred,labels_train)
            vd_pred,losses,p= sess.run([output,loss,dist],feed_dict = {x1 : valid_q1,x2 : valid_q2,y:labels_valid})
	    vd_acc = accuracy(vd_pred,labels_valid)
	    vd_acc_6t = accuracy(vd_pred,labels_valid,t=0.6)
	    vd_acc_8t = accuracy(vd_pred,labels_valid,t=0.8)
	    vd_acc_2t = accuracy(vd_pred,labels_valid,t=0.2)
            vd_acc_5t = accuracy(vd_pred,labels_valid,t=0.5)
            vd_acc_3t = accuracy(vd_pred,labels_valid,t=0.3)
            vd_acc_4t = accuracy(vd_pred,labels_valid,t=0.4)
            vd_acc_9t = accuracy(vd_pred,labels_valid,t=0.9)
	    print "Steps: {0}, Validation accuracy :0.7= {1}% , 0.6= {4}%, 0.8= {5}%, 0.2= {6}%, 0.3= {7}%, 0.4= {8}%, 0.9= {9}%, 0.5= {10}%, Validation loss: {2}%, Training loss: {3}".format(step,vd_acc,losses,l,vd_acc_6t,vd_acc_8t,vd_acc_2t,vd_acc_3t,vd_acc_4t,vd_acc_9t,vd_acc_5t)
	#print step
    	step+=1
    test_predictions = predictions(sess.run(output,feed_dict = {x1 : test_q1_sentence_array,
                                                  x2 : test_q2_sentence_array}))


# In[35]:

test_dataframe = pd.DataFrame()

print "done"
# In[34]:

test_dataframe['test_id'] = range(test_q1_sentence_array.shape[0])
test_dataframe['is_duplicate'] = list(test_predictions)
test_dataframe.to_csv('/home/sayon/submission_file_sayon.csv',index = False)
print "csv generated"
