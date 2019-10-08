#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from utils.util import *
from config import *
from model.Base_Model import *

class LSTMCRF(Base_Model):
	def __init__(self, args):
		super(LSTMCRF,self).__init__(args)
		if args.algorithm=='LSTMCRF':
			self.build()

	def build(self):
		self.mix()
		if self.args.mode in ['train','tune']:
			self.loss_layer()

	def mix(self):
		char_embed=self.char_embedding_layer_lstm()
		word_embed=self.word_embedding_layer()
		self.word_embed=word_embed
		concat_embed=tf.concat([word_embed,char_embed],axis=-1)
		self.concat_embed=concat_embed
		concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
		encode,cell,of,ob=self.lstm(concat_embed,hidden=self.hidden_dim)
		encode=tf.nn.dropout(encode,self.dropout_keep_prob)
		output=self.dense(encode,self.output_class_num,'dense2',linear=True)*self.mask_x
		self.output=output

		with tf.variable_scope('crf'):
			dim_crf1=self.output_class_num
			crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1=self.crf_loss_builtin(
				output
				,self.y_tag_sparse
				,dim_crf1)

			loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
			tf.summary.scalar('crf_loss',loss)
			self.loss=loss

			self.viterbi_sequence=self.crf_predict(output
				,W_crf_lstm1
				,b_crf_lstm1
				,transition_params_lstm1
				,dim_crf1)
		# loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_output,logits=output)
		# self.loss=loss


	def lstm(self,field,hidden=300, name='lstm'):
		with tf.variable_scope(name):
			s=tf.shape(field)
			output,cell,of,ob=self.blstm_layer('blstm',field,
				s,
				hidden,
				self.batch_actual_length)
			
			lower_lstm=output
			of=of
			ob=ob
			return lower_lstm,cell,of,ob


	def loss_layer(self):
		with tf.variable_scope('loss_layer'): 
			decay_learning_rate=self.lr
			tf.summary.scalar('lr',decay_learning_rate)
			self.train_op=tf.train.AdamOptimizer(decay_learning_rate).minimize(self.loss)


	def crf_predict(self,inp,W_crf,b_crf,transition_params,concate_size):
		unary_score=tf.reshape(inp,[-1,self.max_length,self.output_class_num])
		viterbi_sequence,viterbi_score=tf.contrib.crf.crf_decode(unary_score,transition_params,self.batch_actual_length)
		return viterbi_sequence


	def prediction_4_test(self,sess,feed_dict):
		"""
		Args:
			x: input sample
		
		Return:
			prediction: the ix of class
		"""
		# new operation add into the graph in for loop is not accepted memory leak
		[prediction]=sess.run([self.viterbi_sequence],feed_dict=feed_dict)
		
		x=feed_dict[self.x_data]
		batch_actual_length=feed_dict[self.batch_actual_length]
		prediction=np.reshape(prediction,[-1,feed_dict[self.max_length]])
		predictionx=[[e for i,e in enumerate(x) if i<batch_actual_length[j]]for j,x in enumerate(prediction)]
		return predictionx,prediction

	def attend(self,x,y,mask_x,mask_y,tp='tanh',name='attend',mask_self=False,dim=None,use_localm=False,local_mask=None):
		with tf.variable_scope(name):
			
			def _3dimdot2dim(_3d,_2d):
				s=tf.shape(_3d)
				return tf.reshape(tf.matmul(tf.reshape(_3d,[-1,s[-1]]),_2d),s)

			if dim is None:
				dim=self.word_embedding_size
			if tp=='tanh':
				w1=tf.get_variable(name='w1',
					dtype=tf.float32,
					shape=[dim,dim],
					initializer=self.initializer)
				if mask_self:
					w2=w1
				else:
					w2=tf.get_variable(name='w2',
						dtype=tf.float32,
						shape=[dim,dim],
						initializer=self.initializer)
				w=tf.get_variable(name='w',
					dtype=tf.float32,
					shape=[1,dim],
					initializer=self.initializer)
				
				weight=tf.expand_dims(_3dimdot2dim(x,w1),axis=2)+tf.expand_dims(_3dimdot2dim(y,w2),axis=1)
				# s=tf.shape(weight)
				weight=tf.nn.tanh(weight)
				weight=weight*tf.expand_dims(tf.expand_dims(w,axis=0),axis=0)
				weight=tf.reduce_sum(weight,axis=-1)

				weight_y=tf.exp(weight)*mask_x*tf.transpose(mask_y,perm=[0,2,1])
				
				if mask_self:
					self.weight_y=weight_y
				alpha_y=weight_y/(tf.reduce_sum(weight_y,axis=-1,keepdims=True)+1e-12)
				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_y=alpha_y*m_ex

				tf.summary.image('alpha_y',tf.expand_dims(tf.expand_dims(alpha_y,axis=-1)[0],dim=0))

				weight_x=tf.exp(tf.transpose(weight,perm=[0,2,1]))*mask_y*tf.transpose(mask_x,perm=[0,2,1])
				if mask_self:
					self.weight_x=weight_x
				alpha_x=weight_x/(tf.reduce_sum(weight_x,axis=-1,keepdims=True)+1e-12)
				if mask_self:
					sa=tf.shape(alpha_x)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_x=alpha_x*m_ex
				# self.alpha_x=alpha_x
				tf.summary.image('alpha_x',tf.expand_dims(tf.expand_dims(alpha_x,axis=-1)[0],dim=0))

			elif tp=='ny':
				# ====== new york attend =====
				if use_localm:
					weight=tf.matmul(x,tf.transpose(y,perm=[0,2,1]))*local_mask
				else:
					weight=tf.matmul(x,tf.transpose(y,perm=[0,2,1]))
				
				weight_y=tf.exp(weight-tf.reduce_max(weight,axis=2,keepdims=True))*mask_x*tf.transpose(mask_y,perm=[0,2,1])

				alpha_y=weight_y/(tf.reduce_sum(weight_y,axis=-1,keepdims=True)+1e-12)

				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_y=alpha_y*m_ex
				tf.summary.image('alpha_y',tf.expand_dims(tf.expand_dims(alpha_y,axis=-1)[0],axis=0))

				weight_x=tf.exp(tf.transpose(weight-tf.reduce_max(weight,axis=1,keepdims=True),perm=[0,2,1]))*mask_y*tf.transpose(mask_x,perm=[0,2,1])
				alpha_x=weight_x/(tf.reduce_sum(weight_x,axis=-1,keepdims=True)+1e-12)

				if mask_self:
					sa=tf.shape(alpha_x)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_x=alpha_x*m_ex

				# self.alpha_x=alpha_x
				tf.summary.image('alpha_x',tf.expand_dims(tf.expand_dims(alpha_x,axis=-1)[0],dim=0))
			
			elif tp=='sf':
				w1=tf.get_variable(name='w1',
					dtype=tf.float32,
					shape=[dim,dim],
					initializer=self.initializer)
				w2=w1

				d1=_3dimdot2dim(x,w1)
				d2=_3dimdot2dim(y,w2)
				weight=tf.matmul(d1,tf.transpose(d2,perm=[0,2,1]))/tf.sqrt(float(dim))
				weight_y=tf.exp(weight)*mask_x*tf.transpose(mask_y,perm=[0,2,1])
				alpha_y=weight_y/(tf.reduce_sum(weight_y,axis=-1,keepdims=True)+1e-12)
				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_y=alpha_y*m_ex
				tf.summary.image('alpha_y',tf.expand_dims(tf.expand_dims(alpha_y,axis=-1)[0],axis=0))
				weight_x=tf.exp(tf.transpose(weight,perm=[0,2,1]))*mask_y*tf.transpose(mask_x,perm=[0,2,1])
				alpha_x=weight_x/(tf.reduce_sum(weight_x,axis=-1,keepdims=True)+1e-12)
				if mask_self:
					sa=tf.shape(alpha_x)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_x=alpha_x*m_ex

				tf.summary.image('alpha_x',tf.expand_dims(tf.expand_dims(alpha_x,axis=-1)[0],dim=0))
				
			elif tp=='cos':
				#======= cosine attent =======
				# def cos_sim(x,y):
			 	#  		normalize_a = tf.nn.l2_normalize(x,axis=-1)        
				# 	normalize_b = tf.nn.l2_normalize(y,axis=-1)
				# 	cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b),axis=-1,keepdims=True)
				# 	return cos_similarity
				
				alpha_y=self.cos_sim(tf.expand_dims(x,2),tf.expand_dims(y,1))*mask_x*tf.transpose(mask_y,perm=[0,2,1])
				
				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_y=alpha_y*m_ex

				alpha_x=tf.transpose(alpha_y,perm=[0,2,1])
				

				tf.summary.image('alpha_y',tf.expand_dims(tf.expand_dims(alpha_y,axis=-1)[0],axis=0))
				tf.summary.image('alpha_x',tf.expand_dims(tf.expand_dims(alpha_x,axis=-1)[0],dim=0))

			elif tp=='mp':
				l=50
				w1=tf.get_variable(name='w1',
					dtype=tf.float32,
					shape=[l, dim],
					initializer=self.initializer)

				w2=tf.get_variable(name='w2',
					dtype=tf.float32,
					shape=[l, dim],
					initializer=self.initializer)
				mx=tf.expand_dims(x,axis=2)*tf.expand_dims(tf.expand_dims(w1,axis=0),axis=0)
				my=tf.expand_dims(y,axis=2)*tf.expand_dims(tf.expand_dims(w2,axis=0),axis=0)
				alpha_y=self.cos_sim(tf.expand_dims(mx,2),tf.expand_dims(my,1))
				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(tf.expand_dims(m,axis=0),axis=-1)
					alpha_y=alpha_y*m_ex

				x_composition_by_y=tf.reduce_sum(alpha_y,axis=2)*mask_x
				y_composition_by_x=tf.reduce_sum(alpha_y,axis=1)*mask_y
				tf.summary.image('mp_x',tf.expand_dims(tf.expand_dims(x_composition_by_y,axis=-1)[0],axis=0))
				tf.summary.image('mp_y',tf.expand_dims(tf.expand_dims(y_composition_by_x,axis=-1)[0],axis=0))
				return x_composition_by_y,y_composition_by_x

			elif tp=='linear':
				w1=tf.get_variable(name='w1',
					dtype=tf.float32,
					shape=[dim,dim],
					initializer=self.initializer)

				def _3dimdot2dim(_3d,_2d):
					s=tf.shape(_3d)
					return tf.reshape(tf.matmul(tf.reshape(_3d,[-1,s[-1]]),_2d),s)

				# ======= tanh attend =======
				weight=tf.matmul(_3dimdot2dim(x,w1),tf.transpose(y,perm=[0,2,1]))
				# weight=tf.nn.tanh(weight)

				weight_y=tf.exp(weight)*mask_x*tf.transpose(mask_y,perm=[0,2,1])
				self.weight_y=weight_y
				alpha_y=weight_y/(tf.reduce_sum(weight_y,axis=-1,keepdims=True)+1e-12)
				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_y=alpha_y*m_ex

				tf.summary.image('alpha_y',tf.expand_dims(tf.expand_dims(alpha_y,axis=-1)[0],dim=0))

				weight_x=tf.exp(tf.transpose(weight,perm=[0,2,1]))*mask_y*tf.transpose(mask_x,perm=[0,2,1])
				self.weight_x=weight_x
				alpha_x=weight_x/(tf.reduce_sum(weight_x,axis=-1,keepdims=True)+1e-12)
				if mask_self:
					sa=tf.shape(alpha_x)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_x=alpha_x*m_ex
				# self.alpha_x=alpha_x
				tf.summary.image('alpha_x',tf.expand_dims(tf.expand_dims(alpha_x,axis=-1)[0],dim=0))

			elif tp=='abcnn':
				w1=tf.get_variable(name='w1',
					dtype=tf.float32,
					shape=[dim,dim],
					initializer=self.initializer)
				if mask_self:
					w2=w1
				else:
					w2=tf.get_variable(name='w2',
						dtype=tf.float32,
						shape=[dim,dim],
						initializer=self.initializer)
				weight=tf.expand_dims(x,axis=2)-tf.expand_dims(y,axis=1)
				dist=tf.sqrt(tf.reduce_sum(tf.square(weight),axis=-1))*mask_x*tf.transpose(mask_y,perm=[0,2,1])
				alpha_y=1.0/(1.0+dist)*mask_x*tf.transpose(mask_y,perm=[0,2,1])

				if mask_self:
					sa=tf.shape(alpha_y)
					m=tf.diag(tf.ones(sa[1]))*-1.0+1.0
					m_ex=tf.expand_dims(m,axis=0)
					alpha_y=alpha_y*m_ex

				alpha_x=tf.transpose(alpha_y,perm=[0,2,1])

				tf.summary.image('alpha_y',tf.expand_dims(tf.expand_dims(alpha_y,axis=-1)[0],axis=0))
				tf.summary.image('alpha_x',tf.expand_dims(tf.expand_dims(alpha_x,axis=-1)[0],dim=0))
			
			if 's0' in name:
				self.alpha_x=alpha_x
				self.alpha_y=alpha_y

		# y in the eyes of x, attend x
		x_composition_by_y=tf.reduce_sum(tf.expand_dims(y,1)*tf.expand_dims(alpha_y,-1),2)*mask_x
		y_composition_by_x=tf.reduce_sum(tf.expand_dims(x,1)*tf.expand_dims(alpha_x,-1),2)*mask_y

		return x_composition_by_y,y_composition_by_x