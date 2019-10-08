#author: Hongyin Zhu
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from utils.util import *
from model.LSTMCRF import *

class Joint(LSTMCRF):
	def __init__(self, args):
		super(Joint,self).__init__(args)
		if args.algorithm=='Joint':
			self.build()

	def build(self):
		self.mix()
		if self.args.mode in ['train','tune']:
			self.loss_layer()

	def mix(self):
		encode1=self.mix_lstm1('net1')
		encode2=self.mix_lstm1('net2')

		# this place is easy to mistake
		if self.args.mask_net1==1:
			encode1=encode1*0.0
		if self.args.gradient_stop_net1==1:
			encode1=tf.stop_gradient(encode1)

		if self.args.mask_net2==1:
			encode2=encode2*0.0
		if self.args.gradient_stop_net2==1:
			encode2=tf.stop_gradient(encode2)
		
		output=[]
		if self.args.gradient_stop_net1 * self.args.gradient_stop_net2:
			tf.summary.image('encode1',tf.expand_dims(tf.expand_dims(encode1,axis=-1)[0],dim=0))
			tf.summary.image('encode2',tf.expand_dims(tf.expand_dims(encode2,axis=-1)[0],dim=0))
			# tf.summary.image('encode3',tf.expand_dims(tf.expand_dims(encode3,axis=-1)[0],dim=0))
			# tf.summary.image('encode4',tf.expand_dims(tf.expand_dims(encode4,axis=-1)[0],dim=0))
			encode=tf.concat([encode1,encode2],axis=-1)
			
			encode=self.dense(encode,100,'dense3-1',linear=False,bias=True)*self.mask_x
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)
			output=self.dense(encode,self.output_class_num,'dense3-2',linear=True,bias=True)*self.mask_x
		
		else:
			# output1 : 50
			output1=self.dense(encode1,self.output_class_num,'l-left',linear=True,bias=True)*self.mask_x
			output2=self.dense(encode2,self.output_class_num,'l-right',linear=True,bias=True)*self.mask_x
			# output3=self.dense(encode3,self.output_class_num,'l-spanish',linear=True,bias=True)*self.mask_x
			# output4=self.dense(encode4,self.output_class_num,'l-dutch',linear=True,bias=True)*self.mask_x
			
			if self.args.mask_net1==0:
				output=output1
			if self.args.mask_net2==0:
				output=output2

			# output=output1

			# elif self.mask3==0:
			# 	output=output3
			# elif self.mask4==0:
			# 	output=output4

		self.output=output
		with tf.variable_scope('crf'):
			dim_crf1=self.output_class_num
			crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1= \
			self.crf_loss_builtin(
				output
				,self.y_tag_sparse
				,dim_crf1
				,test=False
				,max_length=self.max_length
				,batch_actual_length=self.batch_actual_length)

			loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
			tf.summary.scalar('crf_loss',loss)
			self.loss=loss

			self.viterbi_sequence=self.crf_predict(output
				,W_crf_lstm1
				,b_crf_lstm1
				,transition_params_lstm1
				,dim_crf1)
		print('class-num...',self.output_class_num)


	# These sample blocks will be further organized, and this toolkit will provide more easy-to-use blocks.
	def mix_lstm1_oov(self,name='baseline1'):
		with tf.variable_scope(name):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm()
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm1(self,name='baseline1'):
		with tf.variable_scope(name):
			word_embed=self.word_embedding_layer_base()
			char_embed=self.char_embedding_layer_lstm()
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)*self.mask_x
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode*self.mask_x,self.dropout_keep_prob)
			return encode

	def mix_lstm_random(self,name='baseline1'):
		with tf.variable_scope(name):
			concat_embed=self.word_embedding_layer_random()
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode
			

	def mix_stacka(self,name='baseline1'):
		with tf.variable_scope(name):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm()
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			encode=self.cnn_layer(1,600,encode)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_stackb(self,name='baseline1'):
		with tf.variable_scope(name):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_cnn()
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode=self.cnn_layer(1,300,concat_embed)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			encode,cell,_,_=self.lstm(encode,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm2(self):
		with tf.variable_scope('basline_net2'):
			base_embed=self.word_embedding_layer_google()
			word_embed=self.word_embedding_layer_share100(base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm3(self):
		with tf.variable_scope('basline_net3'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm4(self):
		with tf.variable_scope('basline_net4'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			concat_embed=tf.concat([word_embed,char_embed,self.lm],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode


	def att(self):
		with tf.variable_scope('att'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')

			# x_by_y,_=self.attend(
			# 	word_embed
			# 	,word_embed
			# 	,self.mask_x,self.mask_x
			# 	,tp='tanh',name='attendw'
			# 	# ,mask_self=True
			# 	,dim=200
			# 	)

			# x_by_yc,_=self.attend(
			# 	char_embed
			# 	,char_embed
			# 	,self.mask_x,self.mask_x
			# 	,tp='tanh',name='attendc'
			# 	# ,mask_self=True
			# 	,dim=100
			# 	)

			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			

			concat_embed1=self.dense(concat_embed,100,'d1',linear=True,bias=False)
			concat_embed2=self.dense(concat_embed,100,'d2',linear=True,bias=False)
			concat_embed3=self.dense(concat_embed,100,'d3',linear=True,bias=False)
			
			c1,_=self.attend(
				concat_embed1
				,concat_embed1
				,self.mask_x,self.mask_x
				,tp='ny',name='1'
				# ,mask_self=True
				,dim=100
				# ,use_localm=True
				# ,local_mask=self.local_mask10
				)
			c2,_=self.attend(
				concat_embed2
				,concat_embed2
				,self.mask_x,self.mask_x
				,tp='ny',name='2'
				# ,mask_self=True
				,dim=100
				# ,use_localm=True
				# ,local_mask=self.local_mask11
				)
			c3,_=self.attend(
				concat_embed3
				,concat_embed3
				,self.mask_x,self.mask_x
				,tp='ny',name='3'
				# ,mask_self=True
				,dim=100
				# ,use_localm=True
				# ,local_mask=self.local_mask22
				)
			encode=tf.concat([c1,c2,c3],axis=-1)

			encode=tf.nn.dropout(encode,self.dropout_keep_prob)

			encode,_=self.attend(
				encode
				,encode
				,self.mask_x,self.mask_x
				,tp='ny',name='4'
				# ,mask_self=True
				,dim=300
				# ,use_localm=True
				# ,local_mask=self.local_mask11
				)
			# encode=self.dense(encode,600,'dsf',linear=True,bias=False)
			return encode

	def mix_att(self):
		with tf.variable_scope('att_net'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			x_len=tf.cast(tf.expand_dims(self.batch_actual_length,axis=-1),tf.float32)
			# word_embed=self.word_embed
			# char_embed=self.char_embed
			
			x_by_y,_=self.attend(
				word_embed
				,word_embed
				,self.mask_x,self.mask_x
				,tp='tanh',name='attendw'
				# ,mask_self=True
				,dim=200
				)

			x_by_yc,_=self.attend(
				char_embed
				,char_embed
				,self.mask_x,self.mask_x
				,tp='tanh',name='attendc'
				# ,mask_self=True
				,dim=100
				)

			concat_embed=tf.concat([x_by_y,x_by_yc],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)

			# x_by_y,_=self.attend(
			# 	concat_embed
			# 	,concat_embed
			# 	,self.mask_x,self.mask_x
			# 	,tp='ny',name='attendw'
			# 	# ,mask_self=True
			# 	,dim=200
			# 	)

			encode,cell,f,b=self.lstm(concat_embed,hidden=self.hidden_dim,name='lstm1')
			
			# f_lstm,_=self.attend(
			# 	f
			# 	,f
			# 	,self.mask_x,self.mask_x
			# 	,tp='ny',name='lstmf'
			# 	# ,mask_self=True
			# 	,dim=100
			# 	)

			# b_lstm,_=self.attend(
			# 	b
			# 	,b
			# 	,self.mask_x,self.mask_x
			# 	,tp='ny',name='lstmb'
			# 	# ,mask_self=True
			# 	,dim=100
			# 	)
			# encode=tf.concat([f_lstm,b_lstm],axis=-1)

			# encode,_=self.attend(
			# 	tf.layers.dense(encode,200,use_bias=False)
			# 	,tf.layers.dense(encode,200,use_bias=False)
			# 	,self.mask_x,self.mask_x
			# 	,tp='sf',name='lstmb'
			# 	# ,mask_self=True
			# 	,dim=200
			# 	)

			# encode=tf.concat([x_by_y,encode],-1)
			# encode=self.dense(encode,600,'liner2',linear=True,bias=False)*self.mask_x
			
			# encode=tf.layers.dense(encode,300)+tf.layers.dense(concat_embed,300)
			# encode=tf.nn.dropout(encode,self.dropout_keep_prob)
			# encode,cell,_,_=self.lstm(encode,hidden=300,name='lstm2')
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode
			

	def cnn(self,name='cnn_net'):
		with tf.variable_scope(name):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			x_len=tf.cast(tf.expand_dims(self.batch_actual_length,axis=-1),tf.float32)
			
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)

			# x_by_y,_=self.attend(
			# 	word_embed
			# 	,word_embed
			# 	,self.mask_x,self.mask_x
			# 	,tp='ny',name='attendw'
			# 	# ,mask_self=True
			# 	,dim=200
			# 	)

			# x_by_yc,_=self.attend(
			# 	char_embed
			# 	,char_embed
			# 	,self.mask_x ,self.mask_x
			# 	,tp='ny',name='attendc'
			# 	# ,mask_self=True
			# 	,dim=100
			# 	)

			# encode,cell,f,b=self.lstm(concat_embed,hidden=self.hidden_dim,name='lstm1')
			encode=self.cnn_layer(1,concat_embed)
			# encode=self.cnn_layer(2,encode)
			# x_by_y,_=self.attend(
			# 	concat_embed
			# 	,concat_embed
			# 	,self.mask_x,self.mask_x
			# 	,tp='sf',name='dontknowatt'
			# 	# ,mask_self=True
			# 	,dim=300
			# 	)

			encode=tf.concat([encode],-1)

			# encode,cell,f,b=self.lstm(encode,hidden=self.hidden_dim,name='lstm1')
			# encode=self.dense(encode,600,'liner2',linear=True,bias=False)*self.mask_x
			
			# encode=tf.layers.dense(encode,300)+tf.layers.dense(concat_embed,300)
			# encode=tf.nn.dropout(encode,self.dropout_keep_prob)
			# encode,cell,_,_=self.lstm(encode,hidden=300,name='lstm2')
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def att_instance(self):
		with tf.variable_scope('att_instance'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			x_len=tf.cast(tf.expand_dims(self.batch_actual_length,axis=-1),tf.float32)


			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)

			encode,cell,f,b=self.lstm(concat_embed,hidden=self.hidden_dim,name='lstm1')
			
			dim=600
			r=tf.get_variable(name='r',
					dtype=tf.float32,
					shape=[1,dim],
					initializer=self.initializer)

			A=tf.get_variable(name='A',
					dtype=tf.float32,
					shape=[dim,dim],
					initializer=self.initializer)

			ca=tf.matmul(cell,A)
			# (b,600)*(600,1 )=(b,1)
			alpha=tf.matmul(ca,tf.transpose(r))
			alpha=tf.nn.softmax(alpha,axis=0)
			self.batch_att=alpha

			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	