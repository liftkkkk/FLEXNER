## FlexNER
Here is the project [page](https://liftkkkk.github.io/FLEXNER/).

#### As long as this paper is accepted, this toolkit can be downloaded.

### Requirement:

    Python 2  
    Tensorflow: >=1.4  
    
### Usage

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
	        
  For the Baseline model
```
python train.py -a base 
```
For the Joint training
```
python joint.py -a join
```
For the separated training
```
(1) python joint.py -a join -r2 1 [-g2 1]
(2) python joint.py -a join -r1 1 [-g1 1] -mp model_path -m tune
(3) python joint.py -a join -g1 1 -g2 1 -mp model_path -m tune
```
    
### Dataset

+ CoNLL-2003 dataset are listed [here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)  
+ NYT dataset can be [downloaded](https://github.com/shanzhenren/CoType)  

### Addition

3 steps to build a simple NER arch.
```python
class Bi_Stacka(Bi_NER):
    # initialize the constructor
    ...
    
    # defined a arch.
    def mix(self):
        # 1. add the embeddings
        self.base_embed=self.embedding_layer_base()
		
        # 2. define your arch.
        encode1=self.mix_stacka('net1')
        encode2=self.mix_stacka('net2')
        
        # concatenate the vector
         self.encode=tf.concat([encode1,encode2],axis=-1)

        # additional process
    	...
    	
    	# 3. add a crf layer
    	self.crf_layer()
```
### System Overview

The main modules of our system are listed here. The converter module converts the data into the structured form. Then, if using data augmentation, it will add more instances. The structured data is input into the Pre-process module to vectorize them. Next, the vectors are persisted to the database. Alternatively, the real-time process is also okay, but it will add redundant computation. Then, the Post-processing module converts results into user-friendly form. The evaluation module assesses the result.  
<div align="center">
<img src="./icon/ner_pipline.png" width="450" />
</div> 


### Updating...

* 2017-Sep-10, Bi_NER v0.1, initial version
* 2018-Apr-05, Bi_NER v0.2, supproting easily customize arch. and attention mechanism
* 2018-Nov-03, Bi_NER v0.3, supporting different languages ( tested on English, German, Spanish, Dutch) and biomedical domain
