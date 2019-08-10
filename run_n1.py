#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from model.Joint import *
from utils.preprocess import *
from utils.util import *
from utils.general_ops import *
from run.eval import *
import os
import pickle
import time
import argparse
import random
import tensorflow as tf

v=1234
random.seed(v)
np.random.seed(v)
tf.set_random_seed(v)

parser=argparse.ArgumentParser(description='I do not know')
parser.add_argument('-a','--algorithm',default='Joint', help='model name')
parser.add_argument('-m','--mode',default='train', help='restore train tune')
parser.add_argument('-mp','--model_path',default=None, help='load a model')
parser.add_argument('--save_model_dir',default=None, help='')
# parser.add_argument('--save_log_dir',default=None, help='')
parser.add_argument('--lang',default=None, help='language name')
parser.add_argument('--train_h5',default=None, help='train set h5 file')
parser.add_argument('--test_h5',default=None, help='test set h5 file')
parser.add_argument('--test_pkl',default=None, help='test set pkl file')
parser.add_argument('--corpus',default=None, help='corpus name')
parser.add_argument('--results_report',default=None, help='F1 score file')
parser.add_argument('--predict_file',default=None, help='')
parser.add_argument('--epoch',default=100, type=int, help='')
parser.add_argument('--save_step',default=1000, type=int, help='how many step to save a model')
parser.add_argument('--use_random_embed',default=0, type=int, help='')
parser.add_argument('--word_embed_h5',default=None,  help='')
parser.add_argument('--word_embed_voc',default=None, help='')
parser.add_argument('--char_voc',default=None, help='')
parser.add_argument('--build_voc',default=None, help='')

parser.add_argument('-g1','--gradient_stop_net1',type=int,default=0, help='1:True 0:False')
parser.add_argument('-g2','--gradient_stop_net2',type=int,default=0, help='1:True 0:False')
parser.add_argument('-g3','--gradient_stop_net3',type=int,default=0, help='1:True 0:False')
parser.add_argument('-g4','--gradient_stop_net4',type=int,default=0, help='1:True 0:False')

parser.add_argument('-r1','--mask_net1',type=int,default=0, help='1:True 0:False')
parser.add_argument('-r2','--mask_net2',type=int,default=0, help='1:True 0:False')
parser.add_argument('-r3','--mask_net3',type=int,default=0, help='1:True 0:False')
parser.add_argument('-r4','--mask_net4',type=int,default=0, help='1:True 0:False')

args = parser.parse_args()

ev=[]
_dict={}

if args.corpus=='conll2003':
	ev=Eval('../model/ner/testb.hdf5','../model/ner/testb.hdf5.data')
	_dict=load_data_from_h5('../model/ner/train.hdf5')

# german
if args.corpus=='Germanconll2003':
	ev=Eval('../model/ner/german.testb.hdf5','../model/ner/german.testb.hdf5.data')
	_dict=load_data_from_h5('../model/ner/german.train.hdf5')

# spanish
if args.corpus=='spanish':
	ev=Eval('../model/ner/spanish.testb.hdf5','../model/ner/spanish.testb.hdf5.data')
	_dict=load_data_from_h5('../model/ner/spanish.train.hdf5')

# dutch
if args.corpus=='dutch':
	ev=Eval('../model/ner/dutch.testb.hdf5','../model/ner/dutch.testb.hdf5.data')
	_dict=load_data_from_h5('../model/ner/dutch.train.hdf5')

# four
if args.corpus=='four':
	ev=Eval('../model/ner/four.eng.test.hdf5','../model/ner/four.eng.test.hdf5.data')
	# ev=Eval('../model/ner/four.testb.hdf5','../model/ner/four.testb.hdf5.data')
	_dict=load_data_from_h5('../model/ner/four.train.hdf5')

# chinese
if args.corpus=='chinese':
	ev=Eval('../model/ner/chinese.testb.hdf5','../model/ner/chinese.testb.hdf5.data')
	# ev=Eval('../model/ner/four.testb.hdf5','../model/ner/four.testb.hdf5.data')
	_dict=load_data_from_h5('../model/ner/chinese.train.hdf5')

# new
if args.corpus=='new':
	ev=Eval(args.test_h5,args.test_pkl)
	_dict=load_data_from_h5(args.train_h5)

samples=np.array(_dict['sentences_ix'])
sample_amount=samples.shape[0]
sample_shape=samples.shape[1]

if args.algorithm=='LSTMCRF':
	model=LSTMCRF(args)
if args.algorithm=='Joint':
	model=Joint(args)

init=tf.global_variables_initializer()
configs=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
configs.gpu_options.allow_growth=True
sess=tf.Session(config=configs)
sess.run(init)

if args.mode=='restore':
	saver=tf.train.Saver(max_to_keep=None)
	saver.restore(sess,args.model_path)
elif args.mode=='train':
	saver=tf.train.Saver(max_to_keep=None)
elif args.mode=='tune':


cross_lossx=[]
chunk_lossx=[]
crf_lossx=[]
train_loss=[]
iteration_count=1

# os.popen('rm {}/*'.format(config.log_path))
summary_op=tf.summary.merge_all()
writer=tf.summary.FileWriter('{}'.format(args.save_model_dir),sess.graph)
tf.get_default_graph().finalize()

pre_f1=0.0
epoch_ix=-1

t0=time.time()
ind=np.arange(sample_amount)
for epoch in range(0,args.epoch):
	print ('\n############Starting Epoch {} of {}'.format(epoch,args.epoch))
	shuffle_epochs,num_batches=shuffle_epoch_index(sample_amount
	,config.batch_size
	,1)
	batches=shuffle_epochs[0]
	print ('batches {}'.format(len(batches)))
	for ix,batch in enumerate(batches):
		t1=time.time()
		training_dict=batch_feed(batch,_dict,model,config.dropout_keep_prob,epoch)
		training_dict[model.bn]=False
		training_dict[model.dropout_keep_prob]=0.5
		training_dict[model.dropout_keep_prob2]=0.5

		loss,_,summary,output=sess.run([model.loss
			,model.train_op
			,summary_op
			,model.output
			]
			,feed_dict=training_dict)
		t2=time.time()
		# if config.display_real_time:
		s='\rprocess: {:10.2f} loss: {} batch time: {:0.3f} s t time: {:0.3f}h\n'.format(
			float(ix)/float(num_batches)
			,loss
			,t2-t1
			,(t2-t0)/(3600))
		sys.stdout.write(s)
		sys.stdout.flush()

		summary_nums=(iteration_count, epoch+1, ix+1, num_batches+1, loss )
		# if (ix+1)==len(batches):
		# if ix%args.save_step==0 and ix>1:
			# save_loss_model(args,epoch,summary_nums,sess,saver)
			
			# f1,recall,precision=ev.f1_score(config,model,sess)
			# nhs='epoch {}\tf1 score {}\trecall {}\tprecision {}\n'.format(epoch,f1,recall,precision)
			# with open(args.results_report,'a') as m:
			# 	m.write(nhs)

			# if f1>pre_f1:
			# 	pre_f1=f1
			# 	print ('new highest f1 score {}\trecall {}\tprecision {} \n'.format(f1,recall,precision))
		writer.add_summary(summary,iteration_count)
		iteration_count+=1
	save_loss_model(args,epoch,summary_nums,sess,saver)
