#author: Hongyin Zhu
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from utils.util import *
from config import *
# from service.process import *
from utils.load_embeddings import *

class Base_Model(load_embeddings):
	
	def __init__(self,args):
		"""
		Args:
			training_seq_steps: the step number of a sequence
		"""
		# print '=========transfer model gan reconstruct outside========='
		super(Base_Model,self).__init__(args)

		# self.train=train
		# self.process=Process()
		# cause the model fail to train because you overlap the superclass self.config
		config=Config()
		# config=self.config
		# self.config=config
		self.word_embed,self.char_embed,self.vocab,self.charcab=self.load_glove_vocab(pre_char=False)
		# self.google_embed,_,self.g_vocab,_=self.load_google_vocab()
		self.build_vocab=self.load_buildall_vocab()
		# print google_embed.shape
		self.vocab_size=self.word_embed.shape[0]
		print('vocab_size...',self.vocab_size)
		self.charcab_size=len(self.charcab)
		self.build_vocab_size=len(self.build_vocab)
		print('build_vocab_size...',self.build_vocab_size)
		# self.google_size=self.google_embed.shape[0]
		
		self.learning_rate=config.learning_rate
		# self.training=train
		self.hidden_dim=config.hidden_dim
		self.training_seq_steps=config.training_seq_steps
		self.lr_decay_factor=config.lr_decay_factor
		self.dropout_keep_prob=config.dropout_keep_prob
		self.batch_size=config.batch_size
		self.test_batch_size=config.test_batch_size
		self.word_embedding_size=config.word_embedding_size
		self.output_class_num=config.output_class_num
		self.config_folder=config.config_folder
		# self.logs=config.logs
		# they are in the same dir
		self.log_path=self.args.save_model_dir
		self.char_dim=config.dim_char
		self.hidden_size_char=config.hidden_size_char
		self.word_length=config.max_word_length

		self.char_cnn_filter_size=[3]
		self.char_cnn_num_filters=100


		initializer = tf.contrib.layers.xavier_initializer()
		self.initializer=initializer
		self.regularizer=tf.contrib.layers.l2_regularizer(0.001)

		self.add_palceholders()
		# self.build()

	def add_palceholders(self):
		self.lr=tf.placeholder(dtype=tf.float32,shape=(),name='lr')
		self.x_data = tf.placeholder(dtype=tf.int32, shape=[None, None],name='x_data')
		self.x_data_google = tf.placeholder(dtype=tf.int32, shape=[None, None],name='x_data_google')
		self.x_data_random = tf.placeholder(dtype=tf.int32, shape=[None, None],name='r_data_google')
		
		self.y_output = tf.placeholder(dtype=tf.int32, shape=[None, None,self.output_class_num],name='y_output')
		self.y_tag_sparse=tf.placeholder(dtype=tf.int32, shape=[None,None],name='y_tag_sparse')
		self.dropout_keep_prob=tf.placeholder(dtype=tf.float32,shape=(),name='dropout_keep_prob')
		self.dropout_keep_prob2=tf.placeholder(dtype=tf.float32,shape=(),name='dropout_keep_prob2')
		self.batch_actual_length=tf.placeholder(dtype=tf.int32,shape=[None],name='batch_actual_length')
		self.batch_actual_word_length=tf.placeholder(tf.int32,[None,None],'batch_actual_word_length')
		self.bn=tf.placeholder(dtype=tf.bool,shape=(),name='phase')

		self.char_ids=tf.placeholder(name='char_ids'
			,dtype=tf.int32
			,shape=[None,None,self.word_length])

		# self.mask_x=tf.expand_dims(self.mask,axis=-1)
		self.max_length=tf.placeholder(dtype=tf.int32,shape=(),name='max_length')
		self.mask_x=tf.expand_dims(
			tf.sequence_mask(self.batch_actual_length
				,self.max_length
				,dtype=tf.float32
				,name='sequence_mask')
				,axis=-1)

		self.local_mask10=tf.placeholder(dtype=tf.float32,shape=[None,None],name='local_mask10')
		self.local_mask10_=tf.expand_dims(self.local_mask10,axis=0)

		self.local_mask11=tf.placeholder(dtype=tf.float32,shape=[None,None],name='local_mask10')
		self.local_mask11_=tf.expand_dims(self.local_mask11,axis=0)

		self.local_mask22=tf.placeholder(dtype=tf.float32,shape=[None,None],name='local_mask10')
		self.local_mask22_=tf.expand_dims(self.local_mask22,axis=0)

		# self.google_embed=tf.placeholder(dtype=tf.float32, shape=[13945, 300],name='google_embed')


	def char_embedding_layer_lstm(self,name='char_embedding'):
		with tf.variable_scope(name):
			# with tf.device('/device:GPU:0'):
			char_embedding=tf.get_variable(name='_char_embedding',
				dtype=tf.float32,
				shape=[self.charcab_size,self.char_dim],
				initializer=self.initializer)
			# tf.summary.histogram()
			# char_embedding=tf.Variable(self.char_embed
			# 	,name='_char_embedding',
			# 	dtype=tf.float32,
			# 	trainable=False)
			self.char_embedding=char_embedding
			tf.summary.histogram('char_embed',char_embedding)
			word_rpt=tf.nn.embedding_lookup(char_embedding,
				self.char_ids,name='word_rpt')

			s=tf.shape(word_rpt)
			# d=tf.shape(self.word_actual_length)

			word_lengths = tf.reshape(self.batch_actual_word_length, shape=[s[0]*s[1]])

			_,output,_,_=self.blstm_layer('char',word_rpt,
				[s[0]*s[1],s[2],self.char_dim],
				self.hidden_size_char,
				word_lengths)
			# shape = (batch size, max sentence length, char hidden size)
			output = tf.reshape(output,
				shape=[s[0], s[1], 2*self.hidden_size_char])
			concat=output
			dim=2*self.hidden_size_char
			pool_rhp=tf.reshape(concat,[s[0],s[1],dim])
		return pool_rhp

	def cnn_layer(self,i,dim,inp):
		with tf.variable_scope('cnn{}'.format(i)):
			word=inp
			s=tf.shape(word)
			# constant dim length of a word
			# dim=dim

			# sent_conv: (50, 150, 220, 50)
			# you can use 'VALID' because the width is full the length, no need worry miss position
			# if you use 'SAME', the strid step must be 220, because if not, it will pad 0 the result is same shape with input
			cnn_filter_size=[2,3,5]
			num_filters=300
			pool_output=[]
			for i, filter_width in enumerate(cnn_filter_size):
				with tf.variable_scope('channel-width-{}'.format(filter_width)):
					
					filter_shape=[filter_width,dim,1,num_filters]
					sent_conv=self.conv_layer('sent_conv-{}'.format(i),
						word,
						[s[0],s[1],s[2],1],
						filter_shape,
						[1,1,dim,1],
						'relu',
						padding='SAME',
						BN=False,phase=self.bn)

					# # pool_len=self.sent_length-filter_width+1
					sent_conv=tf.transpose(sent_conv,perm=[0,1,3,2])

					pool=self.pool_layer('pool1',
						sent_conv,
						[1,1,3,1],
						[1,1,3,1],
						padding='VALID')

					pool=tf.transpose(pool,perm=[0,1,3,2])

					# pool=sent_conv

					pool_output.append(pool)

			concat=tf.concat(pool_output,axis=-1)
			self.cnnd=concat
			# s=tf.shape(concat)
			# dim_out=num_filters*len(cnn_filter_size)
			sent_rhp=tf.squeeze(concat,2)*self.mask_x
			# sent_rhp=tf.reshape(concat,[s[0],s[1],600])
			return sent_rhp


	def char_embedding_layer_cnn(self,name='char_embedding'):
		with tf.variable_scope(name):
			with tf.device('/cpu:0'):
				char_embedding=tf.get_variable(name='_char_embedding',
					dtype=tf.float32,
					shape=[self.charcab_size,self.char_dim],
					initializer=self.initializer)
				tf.summary.histogram('char_embed',char_embedding)
				# self.embed_list=[]
				# # self.embed_list.append((char_embedding,self.model_dir+'/char.meta'))
				word_rpt=tf.nn.embedding_lookup(char_embedding
					,self.char_ids
					,name='word_rpt')

				s=tf.shape(word_rpt)
			# word_rptr=tf.reshape(word_rpt,shape=[s[0]*s[1],-1,self.char_dim])
			word_rptr=tf.reshape(word_rpt,shape=[s[0]*s[1],s[2]*self.char_dim,1,1])
			# word_rptr_conv=tf.reshape()

			pool_output=[]
			num_filters=self.char_cnn_num_filters
			for i, filter_width in enumerate(self.char_cnn_filter_size):
				with tf.variable_scope('char-conv-{}'.format(i)):
					
					filter_shape=[filter_width,self.char_dim,1,num_filters]
					sent_conv=self.conv_layer('char_channel-{}'.format(i),
						word_rptr,
						[s[0]*s[1],s[2],self.char_dim,1],
						filter_shape,
						[1,1,1,1],
						'relu',
						padding='VALID',
						BN=True,phase=self.bn)

					pool_len=self.word_length-filter_width+1

					pool=self.pool_layer('pool1',
						sent_conv,
						[1,pool_len,1,1],
						[1,1,1,1],
						padding='VALID')

					pool_output.append(pool)

			concat=tf.concat(pool_output,axis=-1) 
			
			dim_out=num_filters*len(self.char_cnn_filter_size)
			pool_rhp=tf.reshape(concat,[s[0],s[1],dim_out])
		return pool_rhp

	def word_embedding_layer_base(self,name='word_embedding'):
		with tf.variable_scope(name):
			# with tf.device('/device:GPU:0'):
			if self.args.use_random_embed==1:
				word_embedding=tf.get_variable(name='_word_embedding',
					dtype=tf.float32,
					shape=[self.vocab_size, 200],
					initializer=self.initializer)
			else:
				# word_embedding=tf.Variable(self.word_embed,
				# 	name='_word_embedding',
				# 	dtype=tf.float32,
				# 	trainable=True)
				word_embedding=tf.get_variable(name='_word_embedding',
					dtype=tf.float32,
					shape=[self.vocab_size, 300],
					initializer=tf.constant_initializer(self.word_embed,dtype=tf.float32),
					trainable=False)

			# self.word_embedding=word_embedding
			# tf.summary.histogram('word_embed',word_embedding)
			# # self.embedding_init = tf.assign(word_embedding,self.pretrained)
			
			field=tf.nn.embedding_lookup(word_embedding,
				self.x_data,name='word')
			# *self.mask_x
			return field

	def word_embedding_layer_share(self,field,name='word_embedding'):
		with tf.variable_scope(name):
			r=self.word_embedding_layer_random()

			unk=tf.equal(self.x_data, 1)
			unk=tf.expand_dims(tf.cast(unk,tf.float32),axis=-1)
			self.unk=unk
			r=r*unk

			k=tf.greater(self.x_data, 1)
			k=tf.expand_dims(tf.cast(k,tf.float32),axis=-1)
			field=field*k
			
			field=field+r
			# field=tf.concat([field,r],-1)
			# print 'no using dynamic embedding...'
			return field

	def word_embedding_layer_random(self):
		# with tf.device('/device:GPU:0'):
		with tf.variable_scope('random_embedding'):
			# goolge_embed=tf.constant(self.google_embed,dtype=tf.float32)
			# word_embedding=tf.get_variable(name='_word_embedding',
			# 	dtype=tf.float32,
			# 	shape=[self.vocab_size, 300],
			# 	initializer=tf.constant_initializer(self.word_embed))

			word_embedding=tf.get_variable(name='_word_embedding',
				dtype=tf.float32,
				shape=[self.build_vocab_size, 200],
				initializer=self.initializer)
			
			field=tf.nn.embedding_lookup(word_embedding,
				self.x_data,name='word')*self.mask_x
			return field

	def word_embedding_layer_share100(self,field,name='word_embedding'):
		with tf.variable_scope(name):
			r=self.word_embedding_layer_random100()
			# tf.matmul(tf.reshape())

			unk=tf.equal(self.x_data, 1)
			unk=tf.expand_dims(tf.cast(unk,tf.float32),axis=-1)
			self.unk=unk
			r=r*unk

			k=tf.greater(self.x_data, 1)
			k=tf.expand_dims(tf.cast(k,tf.float32),axis=-1)
			field=field*k
			
			field=field+r
			return field


	def word_embedding_layer(self,name='word_embedding'):
		with tf.variable_scope(name):
			# with tf.device('/cpu:0'):
			word_embedding=tf.Variable(self.word_embed
				,name='_word_embedding',
				dtype=tf.float32,
				trainable=False)
			self.word_embedding=word_embedding
			# tf.summary.histogram('word_embed',word_embedding)
			# # self.embedding_init = tf.assign(word_embedding,self.pretrained)
			
			field=tf.nn.embedding_lookup(word_embedding,
				self.x_data,name='word')*self.mask_x


			# field_=self.highwary_layer(field,name='glove')

			# g_field_=self.highwary_layer(g_field,name='google')

			r=self.word_embedding_layer_random()
			# tf.matmul(tf.reshape())

			unk=tf.equal(self.x_data, 1)
			unk=tf.expand_dims(tf.cast(unk,tf.float32),axis=-1)
			self.unk=unk
			r=r*unk

			k=tf.greater(self.x_data, 1)
			k=tf.expand_dims(tf.cast(k,tf.float32),axis=-1)
			field=field*k
			
			field=field+r

			# g_field=self.word_embedding_layer_google()

			# m=tf.layers.dense(g_field,200,use_bias=False)
			# z=tf.nn.sigmoid(tf.layers.dense(tf.nn.tanh(tf.layers.dense(m,200,use_bias=False)  \
			# 	+tf.layers.dense(field,200,use_bias=False)),200,use_bias=False))
			# field=z*field+(1-z)*m

			return field

	def word_embedding_layer_google(self):
		with tf.device('/cpu:0'):
			with tf.variable_scope('google_embedding'):
				# goolge_embed=tf.constant(self.google_embed,dtype=tf.float32)

				word_embedding=tf.Variable(self.google_embed,
					# shape=[3000002, 300]
					name='_word_embedding',
					dtype=tf.float32,
					trainable=False)

				# word_embedding=tf.get_variable(name='_word_embedding',
				# 	dtype=tf.float32,
				# 	shape=[13945, 300],
				# 	trainable=False,
				# 	initializer=self.initializer)
				# word_embedding.assign(self.google_embed)

				field=tf.nn.embedding_lookup(word_embedding,
					self.x_data_google,name='field1')*self.mask_x
				return field

	

	def word_embedding_layer_random100(self):
		with tf.device('/cpu:0'):
			with tf.variable_scope('random_embedding'):
				# goolge_embed=tf.constant(self.google_embed,dtype=tf.float32)

				word_embedding=tf.get_variable(name='_word_embedding',
					dtype=tf.float32,
					shape=[self.build_vocab_size, 50],
					initializer=self.initializer)
				# word_embedding.assign(self.google_embed)

				field=tf.nn.embedding_lookup(word_embedding,
					self.x_data_random,name='field1')*self.mask_x
				return field

	def blstm_layer(self,name,input,input_reshape,hidden_state,sequence_length):
		with tf.variable_scope(name):
			input_rhp=tf.reshape(input,input_reshape)
			cell_fw = tf.contrib.rnn.LSTMCell(hidden_state,
				state_is_tuple=True)
			cell_bw = tf.contrib.rnn.LSTMCell(hidden_state,
				state_is_tuple=True)

			# cell_fw=tf.contrib.rnn.AttentionCellWrapper(cell_fw,attn_length=10)
			# cell_bw=tf.contrib.rnn.AttentionCellWrapper(cell_bw,attn_length=10)

			_output = tf.nn.bidirectional_dynamic_rnn(
				cell_fw, cell_bw, input_rhp,
				sequence_length=sequence_length, dtype=tf.float32)
			# read and concat output
			(output_fh, output_bh), ((_, output_fw), (_, output_bw)) = _output
			output= tf.concat([output_fh, output_bh], axis=-1)
			state = tf.concat([output_fw, output_bw], axis=-1)
			return output,state,output_fh,output_bh


	def conv_layer(self,name,input,input_reshape,filter_shape,cnn_strid,active,
		padding='SAME',BN=False,phase=True):
		"""
		SyntaxError: non-default argument follows default argument, any default parametr should behind non-default
		CNN want to channel stride must use a new filter variable transpose the fitler too
		Args:
			filter_shape:[filter_height,filter_width,in_channels,out_channels]
			strid: each dimension of input
			pool_window: each dimension of conv_output

		Exceptions:
			InvalidArgumentError (see above for traceback): Current implementation does not yet support strides in the batch and depth dimensions.
		"""
		with tf.variable_scope(name):

			filter=tf.get_variable('filter',filter_shape,tf.float32,initializer=self.initializer)
			# bias=tf.Variable(tf.constant(0.0,shape=[filter_shape[3]]),name='bias')
			bias=tf.get_variable('bias',[filter_shape[3]],tf.float32,initializer=self.initializer)
			tf.summary.histogram('filter',filter)
			# tf.summary.histogram('bias',bias)
			input=tf.reshape(input,input_reshape)
			conv_out=tf.nn.conv2d(input,filter,strides=cnn_strid,padding=padding)

			conv_out=tf.nn.bias_add(conv_out,bias)

			# conv_out=tf.layers.batch_normalization(conv_out,
			# 	axis=-1,
			# 	center=True,
			# 	scale=True, 
			# 	training=phase,
			# 	name='bn')
			
			if active=='relu':
				activation=tf.nn.relu(conv_out)
			elif active=='tanh':
				activation=tf.nn.tanh(conv_out)
			elif active=='sigmoid':
				activation=tf.nn.sigmoid(conv_out)
			elif active=='leaky_relu':
				activation=tf.nn.leaky_relu(conv_out)
			# tf.summary.histogram('activation',activation)
		return activation


	def pool_layer(self,name,activation,kernel,strides,is_channel_stride=False,padding='SAME'):
		"""

		Exceptions:
			UnimplementedError (see above for traceback): Depthwise max pooling requires the depth window to equal the depth stride, because pooling don't have the overlap region
			depth stride only implement on cpu version
		"""
		with tf.variable_scope(name):
			k=kernel[:3]+[1]
			tmp=strides[:3]+[1]
			pool_out=tf.nn.max_pool(activation,ksize=k,strides=tmp,padding=padding)

			if is_channel_stride:
				strides=[1,1,strides[3],1]
				pool_out=tf.transpose(pool_out,perm=[0,1,3,2])
				# avg_pool
				pool_out=tf.nn.max_pool(pool_out,ksize=strides,strides=strides,padding=padding)
				pool_out=tf.transpose(pool_out,perm=[0,1,3,2])
		return pool_out


	def dense(self,inp,dim,name,activation=tf.nn.relu,linear=False,bias=True,bn=False):
		with tf.variable_scope(name):
			dense=tf.layers.dense(inputs=inp,
				units=dim,
				activation=None,
				kernel_regularizer= None,
				use_bias=bias,
				name='linear')
			# return dense
			if bn:
				# s=tf.shape(dense)
				# dense=tf.expand_dims(dense,axis=2)
				dense_bn=tf.layers.batch_normalization(dense,
					# axis=-1,
					center=True,
					scale=True, 
					training=self.bn,
					name='bn')
				# dense_bn=tf.squeeze(dense_bn,axis=2)
				# dense_bn = tf.keras.layers.BatchNormalization()(dense, training=self.bn)

			else:
				dense_bn=dense

			# dense_bn=tf.contrib.layers.batch_norm(dense,
			# 	center=True,
   # 				scale=True,
   # 				is_training=self.bn
			# 	)
			# dense_bn=dense

			if linear:
				return dense_bn
			dense_ac=activation(dense_bn)
			return dense_ac

	def highwary_layer(self,x,name='highwary'):
		with tf.variable_scope(name):
			s=x.shape
			T=tf.layers.dense(inputs=x,
				units=s[-1],
				activation=tf.nn.sigmoid,
				kernel_regularizer= None,
				use_bias=True,
				name='T')

			H=tf.layers.dense(inputs=x,
				units=s[-1],
				activation=None,
				kernel_regularizer= None,
				use_bias=True,
				name='H')

			y=H*T+(1.0-T)*x
			return y

	# def crf_loss_builtin(self,inp,target,concate_size,test=False):
	# 	"""crf
	# 	Args:
	# 		input: the hidden state to crf the concatenate hidden state [batch_size,training_seq_length,vector_length]
	# 		target: [batch_size,training_seq_length]

	# 	"""
	# 	input_feature_length=concate_size
	# 	# input=input*self.mask_x
	# 	# x_reshape=tf.reshape(input,[-1,input_feature_length])
	# 	W_crf=tf.get_variable('W_crf',[input_feature_length,self.output_class_num]
	# 		,tf.float32,self.initializer,regularizer=self.regularizer)

	# 	b_crf=tf.Variable(tf.constant(0.1,shape=[1,self.output_class_num]),name='b_crf')
		
	# 	# unary_score=tf.matmul(x_reshape,W_crf)
	# 	# # self.unary_score=unary_score
	# 	# # print 'unary_score is {} x_reshape is {} self.W_crf {}'.format(unary_score,x_reshape,self.W_crf)
	# 	# # print 'cancle unary_score mask-----------------------'
	# 	unary_score=tf.reshape(inp,[-1,self.max_length,self.output_class_num])
	# 	self.unary_score=unary_score
	# 	log_likelihood, transition_params=tf.contrib.crf.crf_log_likelihood(unary_score
	# 		,target
	# 		,self.batch_actual_length)
		
	# 	return log_likelihood,transition_params,W_crf,b_crf

	def crf_loss_builtin(self,inp,target,concate_size,test=False,max_length=None,batch_actual_length=None):
		"""crf
		Args:
			input: the hidden state to crf the concatenate hidden state [batch_size,training_seq_length,vector_length]
			target: [batch_size,training_seq_length]

		"""
		input_feature_length=concate_size
		unary_score=tf.reshape(inp,[-1,max_length,self.output_class_num])
		self.unary_score=unary_score
		log_likelihood, transition_params=tf.contrib.crf.crf_log_likelihood(unary_score
			,target
			,batch_actual_length)
		
		return log_likelihood,transition_params,None,None

	def cos_sim(self,x,y):
		normalize_a = tf.nn.l2_normalize(x,axis=-1)        
		normalize_b = tf.nn.l2_normalize(y,axis=-1)
		cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b),axis=-1)
		return cos_similarity