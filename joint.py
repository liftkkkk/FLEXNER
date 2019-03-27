#author: Hongyin Zhu

import tensorflow as tf
import numpy as np
from util import *
from config import *
from lstm_crf_v2 import *

class ATT_J(LSTMCRF):
	def __init__(self, train='train',stop1=0,stop2=0,stop3=0,stop4=0,mask1=0,mask2=0,mask3=0,mask4=0):
		super(ATT_J,self).__init__(train='train')
		# self.train=train
		self.mask1=mask1
		self.mask2=mask2
		self.mask3=mask3
		self.mask4=mask4

		self.stop1=stop1
		self.stop2=stop2
		self.stop3=stop3
		self.stop4=stop4
		# self.mask3,self.stop3=mask3,stop3
		self.build()

	def build(self):
		self.mix()
		if self.train=='train':
			self.loss_layer()

	def mix(self):
		# self.word_embed=self.word_embedding_layer()
		# self.char_embed=self.char_embedding_layer_lstm()
		# assert(self.stop1==1)
		
		# if self.mask1==1 or self.stop1==1:
		# 	self.base_embed=tf.stop_gradient(self.base_embed)
			
		encode1=self.mix_lstm1('net1')
		encode2=self.mix_lstm1('net2')
		# encode3=self.mix_lstm1('net3')
		# encode4=self.mix_lstm1('net4')
		
		# encode1=self.mix_stacka('net1')
		# encode2=self.mix_stacka('net2')

		# encode1=self.mix_stackb()
		# encode2=self.mix_stackb('cnn2')


		# output1=self.dense(encode1,600,'dense11',activation=tf.nn.relu,bias=False)*self.mask_x
		# output2=self.dense(encode2,600,'dense12',activation=tf.nn.relu,bias=False)*self.mask_x

		# output1=tf.nn.dropout(output1,self.dropout_keep_prob)
		# output2=tf.nn.dropout(output2,self.dropout_keep_prob)

		# this place is easy to mistake
		if self.mask1==1:
			encode1=encode1*0.0
		if self.stop1==1:
			encode1=tf.stop_gradient(encode1)

		if self.mask2==1:
			encode2=encode2*0.0
		if self.stop2==1:
			encode2=tf.stop_gradient(encode2)

		# if self.mask3==1:
		# 	encode3=encode3*0.0
		# # you error write self.stop1==1 here
		# if self.stop3==1:
		# 	encode3=tf.stop_gradient(encode3)

		# if self.mask4==1:
		# 	encode4=encode4*0.0
		# if self.stop4==1:
		# 	encode4=tf.stop_gradient(encode4)
		

		output=[]
		if self.stop1 * self.stop2:
			tf.summary.image('encode1',tf.expand_dims(tf.expand_dims(encode1,axis=-1)[0],dim=0))
			tf.summary.image('encode2',tf.expand_dims(tf.expand_dims(encode2,axis=-1)[0],dim=0))
			# tf.summary.image('encode3',tf.expand_dims(tf.expand_dims(encode3,axis=-1)[0],dim=0))
			# tf.summary.image('encode4',tf.expand_dims(tf.expand_dims(encode4,axis=-1)[0],dim=0))
			encode=tf.concat([encode1,encode2],axis=-1)
			print 'hahahahahhha'
			
			encode=self.dense(encode,100,'dense3-1',linear=False,bias=True)*self.mask_x
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)
			output=self.dense(encode,self.output_class_num,'dense3-2',linear=True,bias=True)*self.mask_x
		
		# if self.stop1 * self.stop2 * self.stop3 * self.stop4:
		# 	# encode=tf.concat([encode1,encode2,encode3,encode4],axis=-1)
		# 	print 'hahahahahhha'
		# 	output1=self.dense(encode1,self.output_class_num,'l-english2',linear=False,bias=True)*self.mask_x
		# 	output2=self.dense(encode2,self.output_class_num,'l-german2',linear=False,bias=True)*self.mask_x
		# 	output3=self.dense(encode3,self.output_class_num,'l-spanish2',linear=False,bias=True)*self.mask_x
		# 	output4=self.dense(encode4,self.output_class_num,'l-dutch2',linear=False,bias=True)*self.mask_x
		# 	# encode=tf.concat([output1,output2,output3,output4],axis=1)
		# 	ave=tf.reduce_mean([output2,output3,output4],axis=0)
		# 	x_by_y,_=self.attend(
		# 	output1
		# 	,ave
		# 	,self.mask_x,self.mask_x
		# 	,tp='ny',name='eng-all'
		# 	# ,mask_self=True
		# 	# ,dim=296
		# 	)
		# 	agg=tf.concat([output1,x_by_y],axis=-1)
		# 	# encode=self.dense(encode,100,'dense3-1',linear=False,bias=True)*self.mask_x
		# 	# encode=tf.nn.dropout(encode,self.dropout_keep_prob)
		# 	output=self.dense(agg,self.output_class_num,'dense3-2',linear=True,bias=True)*self.mask_x
		else:
			# output1 : 50
			output1=self.dense(encode1,self.output_class_num,'l-left',linear=True,bias=True)*self.mask_x
			output2=self.dense(encode2,self.output_class_num,'l-right',linear=True,bias=True)*self.mask_x
			# output3=self.dense(encode3,self.output_class_num,'l-spanish',linear=True,bias=True)*self.mask_x
			# output4=self.dense(encode4,self.output_class_num,'l-dutch',linear=True,bias=True)*self.mask_x
			
			if self.mask1==0:
				output=output1
			if self.mask2==0:
				output=output2
			# elif self.mask3==0:
			# 	output=output3
			# elif self.mask4==0:
			# 	output=output4

		self.output=output
		with tf.variable_scope('crf'):
			dim_crf1=self.output_class_num
			crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1=self.crf_loss_builtin(
				output
				,self.y_tag_sparse
				,dim_crf1)
			print 'did not use len in crf'
			loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
			# loss=tf.reduce_sum(-1.0*tf.squeeze(self.batch_att,-1)*crf_log_likelihood_lstm1)
			tf.summary.scalar('crf_loss',loss)
			self.loss=loss

			self.viterbi_sequence=self.crf_predict(output
				,W_crf_lstm1
				,b_crf_lstm1
				,transition_params_lstm1
				,dim_crf1)


	# def mix(self):
	# 	# self.word_embed=self.word_embedding_layer()
	# 	# self.char_embed=self.char_embedding_layer_lstm()

	# 	self.base_embed=self.word_embedding_layer_base()
	# 	encode1=self.mix_lstm1('net1')
	# 	encode2=self.mix_lstm1('net2')
		
	# 	# encode1=self.mix_stacka('net1')
	# 	# encode2=self.mix_stacka('net2')

	# 	# encode1=self.mix_stackb()
	# 	# encode2=self.mix_stackb('cnn2')


	# 	# output1=self.dense(encode1,600,'dense11',activation=tf.nn.relu,bias=False)*self.mask_x
	# 	# output2=self.dense(encode2,600,'dense12',activation=tf.nn.relu,bias=False)*self.mask_x

	# 	# output1=tf.nn.dropout(output1,self.dropout_keep_prob)
	# 	# output2=tf.nn.dropout(output2,self.dropout_keep_prob)

	# 	if self.mask1==1:
	# 		encode1=encode1*0.0
	# 	if self.stop1==1:
	# 		encode1=tf.stop_gradient(encode1)

	# 	if self.mask2==1:
	# 		encode2=encode2*0.0
	# 	if self.stop2==1:
	# 		encode2=tf.stop_gradient(encode2)

	# 	# if self.mask3:
	# 	# 	encode3=encode3*0.0
	# 	# if self.stop3:
	# 	# 	encode3=tf.stop_gradient(encode3)
		

	# 	output=[]
	# 	if self.stop1 * self.stop2:
	# 		print 'hahahahahhha'
	# 		encode=tf.concat([encode1,encode2],axis=-1)
			
	# 		encode=self.dense(encode,600,'dense3-1',linear=False,bias=True)*self.mask_x
	# 		encode=tf.nn.dropout(encode,self.dropout_keep_prob)
	# 		output=self.dense(encode,self.output_class_num,'dense3-2',linear=True,bias=True)*self.mask_x
	# 	else:
	# 		# output1 : 50
	# 		output1=self.dense(encode1,self.output_class_num,'l-1',linear=True,bias=True)*self.mask_x
	# 		output2=self.dense(encode2,self.output_class_num,'l-2',linear=True,bias=True)*self.mask_x
	# 		if self.mask1==0:
	# 			output=output1
	# 		if self.mask2==0:
	# 			output=output2

	# 	self.output=output
	# 	with tf.variable_scope('crf'):
	# 		dim_crf1=self.output_class_num
	# 		crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1=self.crf_loss_builtin(
	# 			output
	# 			,self.y_tag_sparse
	# 			,dim_crf1)
	# 		print 'did not use len in crf'
	# 		loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
	# 		# loss=tf.reduce_sum(-1.0*tf.squeeze(self.batch_att,-1)*crf_log_likelihood_lstm1)
	# 		tf.summary.scalar('crf_loss',loss)
	# 		self.loss=loss

	# 		self.viterbi_sequence=self.crf_predict(output
	# 			,W_crf_lstm1
	# 			,b_crf_lstm1
	# 			,transition_params_lstm1
	# 			,dim_crf1)


	# def mix(self):
	"""
	In NER paper
	"""
	# 	# self.word_embed=self.word_embedding_layer()
	# 	# self.char_embed=self.char_embedding_layer_lstm()
	# 	self.base_embed=self.word_embedding_layer_base()
	# 	# encode1=self.mix_lstm1()
	# 	# encode2=self.mix_lstm1('baseline2')
		
	# 	encode1=self.mix_stacka()
	# 	encode2=self.mix_stacka('cnn2')

	# 	# encode1=self.mix_stackb()
	# 	# encode2=self.mix_stackb('cnn2')


	# 	# output1=self.dense(encode1,600,'dense11',activation=tf.nn.relu,bias=False)*self.mask_x
	# 	# output2=self.dense(encode2,600,'dense12',activation=tf.nn.relu,bias=False)*self.mask_x

	# 	# output1=tf.nn.dropout(output1,self.dropout_keep_prob)
	# 	# output2=tf.nn.dropout(output2,self.dropout_keep_prob)

	# 	if self.mask1:
	# 		encode1=encode1*0.0
	# 	if self.stop1:
	# 		encode1=tf.stop_gradient(encode1)

	# 	if self.mask2:
	# 		encode2=encode2*0.0
	# 	if self.stop2:
	# 		encode2=tf.stop_gradient(encode2)

	# 	# if self.mask3:
	# 	# 	encode3=encode3*0.0
	# 	# if self.stop3:
	# 	# 	encode3=tf.stop_gradient(encode3)
		
	# 	encode=tf.concat([encode1,encode2],axis=-1)

	# 	# =====
	# 	if self.stop1 * self.stop2 or self.mask1+self.mask2==0:
	# 		print 'hahahahahhha'
	# 		# output1=self.dense(encode1,50,'dense21',linear=True,bias=False)*self.mask_x
	# 		# output2=self.dense(encode2,50,'dense22',linear=True,bias=False)*self.mask_x
	# 		# encode=tf.concat([output1,output2],axis=-1)
	# 		# # ===this is the original====
	# 		# output=self.dense(encode,100,'dense3',linear=True,bias=False)*self.mask_x

	# 		# output=self.dense(encode,100,'dense3',linear=True,bias=False)*self.mask_x
			
	# 		encode=self.dense(encode,600,'dense3-1',linear=False,bias=True)*self.mask_x
	# 		encode=tf.nn.dropout(encode,self.dropout_keep_prob)
	# 		output=self.dense(encode,100,'dense3-2',linear=True,bias=False)*self.mask_x
	# 	else:
	# 		output1=self.dense(encode1,50,'o1',linear=True,bias=False)*self.mask_x
	# 		output2=self.dense(encode2,50,'oo',linear=True,bias=False)*self.mask_x
	# 		output=tf.concat([output1,output2],axis=-1)
	# 		output=self.dense(output,100,'dense2',linear=True,bias=False)*self.mask_x

	# 	self.output=output
	# 	with tf.variable_scope('crf'):
	# 		dim_crf1=100
	# 		crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1=self.crf_loss_builtin(
	# 			output
	# 			,self.y_tag_sparse
	# 			,dim_crf1)
	# 		print 'did not use len in crf'
	# 		loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
	# 		# loss=tf.reduce_sum(-1.0*tf.squeeze(self.batch_att,-1)*crf_log_likelihood_lstm1)
	# 		tf.summary.scalar('crf_loss',loss)
	# 		self.loss=loss

	# 		self.viterbi_sequence=self.crf_predict(output
	# 			,W_crf_lstm1
	# 			,b_crf_lstm1
	# 			,transition_params_lstm1
	# 			,dim_crf1)


	# def mix(self):
	# 	self.base_embed=self.word_embedding_layer_base()
	# 	encode1=self.mix_lstm1()

	# 	if self.mask1:
	# 		encode1=encode1*0.0
	# 	if self.stop1:
	# 		encode1=tf.stop_gradient(encode1)

	# 	self.lm=encode1

	# 	encode2=self.mix_lstm4()

	# 	if self.mask2:
	# 		encode2=encode2*0.0
	# 	if self.stop2:
	# 		encode2=tf.stop_gradient(encode2)
		
	# 	if self.stop3==0:
	# 		# =====
	# 		if self.stop1 * self.stop2 or self.mask1+self.mask2==0:
	# 			print 'hahahahahhha'
	# 			output=self.dense(encode2,100,'dense3',linear=True,bias=False)*self.mask_x
	# 		else:
	# 			output=self.dense(encode2,100,'o2',linear=True,bias=False)*self.mask_x
			
	# 		self.output=output
			
	# 		with tf.variable_scope('crf'):
	# 			dim_crf1=100
	# 			crf_log_likelihood_lstm1,transition_params_lstm1,W_crf_lstm1,b_crf_lstm1=self.crf_loss_builtin(
	# 				output
	# 				,self.y_tag_sparse
	# 				,dim_crf1)
	# 			print 'did not use len in crf'
	# 			loss=tf.reduce_mean(-1.0*crf_log_likelihood_lstm1)
	# 			# loss=tf.reduce_sum(-1.0*tf.squeeze(self.batch_att,-1)*crf_log_likelihood_lstm1)
	# 			tf.summary.scalar('crf_loss',loss)
	# 			self.loss=loss

	# 			self.viterbi_sequence=self.crf_predict(output
	# 				,W_crf_lstm1
	# 				,b_crf_lstm1
	# 				,transition_params_lstm1
	# 				,dim_crf1)
	# 	else:
	# 		s=tf.shape(encode1)
	# 		output1=tf.layers.dense(encode1,self.build_vocab_size,use_bias=False)

	# 		output1=output1[:,:-1]
	# 		output1=tf.reshape(output1,[-1,self.build_vocab_size])
	# 		self.output=output1

	# 		target=self.x_data_random[:,1:]
	# 		target=tf.reshape(target,[-1])

	# 		loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([output1], [target], [tf.ones(s[0]*(s[1]-1))])
	# 		loss=tf.reduce_sum(loss)/tf.cast(s[0],tf.float32)
	# 		tf.summary.scalar('lm_loss',loss)
	# 		self.loss=loss

	def mix_lstm1_oov(self,name='baseline1'):
		with tf.variable_scope(name):
			# word_embed=self.base_embed
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm()
			# word_embed=self.word_embed
			# char_embed=self.char_embed
			# concat_embed=word_embed
			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm1(self,name='baseline1'):
		with tf.variable_scope(name):
			concat_embed=self.word_embedding_layer_base()
			# word_embed=self.word_embedding_layer_share(self.base_embed)
			# char_embed=self.char_embedding_layer_lstm()
			# word_embed=self.word_embed
			# char_embed=self.char_embed
			# concat_embed=word_embed
			# concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm_random(self,name='baseline1'):
		with tf.variable_scope(name):
			concat_embed=self.word_embedding_layer_random()
			# word_embed=self.word_embedding_layer_share(self.base_embed)
			# char_embed=self.char_embedding_layer_lstm()
			# word_embed=self.word_embed
			# char_embed=self.char_embed
			# concat_embed=word_embed
			# concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode
			

	def mix_stacka(self,name='baseline1'):
		with tf.variable_scope(name):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm()
			# word_embed=self.word_embed
			# char_embed=self.char_embed

			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
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
			# word_embed=self.word_embed
			# char_embed=self.char_embed

			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
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
			# word_embed=self.word_embed
			# char_embed=self.char_embed

			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm3(self):
		with tf.variable_scope('basline_net3'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			# word_embed=self.word_embed
			# char_embed=self.char_embed

			concat_embed=tf.concat([word_embed,char_embed],axis=-1)
			# self.concat_embed=concat_embed
			concat_embed=tf.nn.dropout(concat_embed,self.dropout_keep_prob)
			encode,cell,_,_=self.lstm(concat_embed,hidden=self.hidden_dim)
			encode=tf.nn.dropout(encode,self.dropout_keep_prob)*self.mask_x
			return encode

	def mix_lstm4(self):
		with tf.variable_scope('basline_net4'):
			word_embed=self.word_embedding_layer_share(self.base_embed)
			char_embed=self.char_embedding_layer_lstm('char_embedding_2')
			# word_embed=self.word_embed
			# char_embed=self.char_embed

			concat_embed=tf.concat([word_embed,char_embed,self.lm],axis=-1)
			# self.concat_embed=concat_embed
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


			# mean=tf.reduce_sum(word_embed,axis=1)/x_len
			# mean=tf.expand_dims(mean,axis=1)
			# s=tf.shape(word_embed)
			# mean_att,_=self.attend(
			# 	word_embed
			# 	,mean
			# 	,self.mask_x,tf.ones([s[0],1,1],dtype=tf.float32)
			# 	,tp='ny',name='attendmean'
			# 	# ,mask_self=True
			# 	,dim=200
			# 	)
			# self.mean_att=mean_att

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

	