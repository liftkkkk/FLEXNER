# Bilateral NER


Requirement:
===
    Python 2  
    Tensorflow: >=1.4  


Usage
===
    usage: joint.py [-h] [-a ALGORITHM] [-m MODE] [-mp MODEL_PATH]
                         [-g1 GRADIENT_STOP_LSTM1] [-g2 GRADIENT_STOP_LSTM2]
                         [-g3 GRADIENT_STOP_LSTM3] [-r1 MASK_LSTM1]
                         [-r2 MASK_LSTM2] [-r3 MASK_LSTM3]

	optional arguments:
	  -h, --help            show this help message and exit
	  -a ALGORITHM, --algorithm ALGORITHM
	                        base att att3 linear join f
	  -m MODE, --mode MODE  restore train tune
	  -mp MODEL_PATH, --model_path MODEL_PATH
	                        ../model/ner/rnn/model-0
	  -g1 GRADIENT_STOP_LSTM1, --gradient_stop_lstm1 GRADIENT_STOP_LSTM1
	                        1:True 0:False
	  -g2 GRADIENT_STOP_LSTM2, --gradient_stop_lstm2 GRADIENT_STOP_LSTM2
	                        1:True 0:False
	  -g3 GRADIENT_STOP_LSTM3, --gradient_stop_lstm3 GRADIENT_STOP_LSTM3
	                        1:True 0:False
	  -r1 MASK_LSTM1, --mask_lstm1 MASK_LSTM1
	                        1:True 0:False
	  -r2 MASK_LSTM2, --mask_lstm2 MASK_LSTM2
	                        1:True 0:False
	  -r3 MASK_LSTM3, --mask_lstm3 MASK_LSTM3
	                        1:True 0:False
  
    
Dataset
===

+ CoNLL-2003 dataset are listed [here](data)  
+ NYT dataset can be [downloaded](https://github.com/shanzhenren/CoType)  


Updating...
===
* 2018-Aug-26, Bi_NER v0.1, initial version
  

