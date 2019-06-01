## FlexNER ([Project Tutorial](https://liftkkkk.github.io/FLEXNER/))

#### As long as this paper is accepted, this toolkit can be downloaded.

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)


FlexNER is a toolkit of neural NER models designed to accelerate ML research. This version of the tutorial requires TensorFlow >=1.4. It is a preview. The detailed descriptions are still in the making.


### Contents

* [Basics](#basics)
  * [Installation](#installation)
  * [Addition](#addition)
    * [Multi-lateral Network](#multi-lateral-network)
    * [Language Correlation](#language-correlation)
    * [Language Interaction](#language-interaction)
* [Suggested Datasets](#suggested-datasets)
  * [CoNLL-2002](#conll-2002)
  * [CoNLL-2003](#conll-2003)
  * [NYT](#nyt)
  * [IOB,IOB2,BIOES Conversion](#iob,iob2,bioes-conversion)
  * [Using Your Data](#using-your-data)
* [Updating](#updating)

    
## Basics
### Installation
```
usage: train.py [-h] [-a ALGORITHM] [-ag AUGMENT] [-m MODE]
                         [-mp MODEL_PATH] [-g1 GRADIENT_STOP_NET1]
                         [-g2 GRADIENT_STOP_NET2] [-g3 GRADIENT_STOP_NET3]
                         [-g4 GRADIENT_STOP_NET4] [-r1 MASK_NET1]
                         [-r2 MASK_NET2] [-r3 MASK_NET3] [-r4 MASK_NET4]

This list provides the options to control the runing status.

optional arguments:
  -h, --help            show this help message and exit
  -a ALGORITHM, --algorithm ALGORITHM
                        Select an algorithm for the model
  -ag AUGMENT, --augment AUGMENT
                        1:True 0:False
  -m MODE, --mode MODE  Select training model. train, restore, tune
  -mp MODEL_PATH, --model_path MODEL_PATH
                        Select the model path
  -g1 GRADIENT_STOP_NET1, --gradient_stop_net1 GRADIENT_STOP_NET1
                        1:True 0:False
  -g2 GRADIENT_STOP_NET2, --gradient_stop_net2 GRADIENT_STOP_NET2
                        1:True 0:False
  -r1 MASK_NET1, --mask_net1 MASK_NET1
                        1:True 0:False
  -r2 MASK_NET2, --mask_net2 MASK_NET2
                        1:True 0:False


```

  For the Baseline model
```
python train.py -a base 
```
For the Joint training
```
python train.py -a join
```
For the separated training
```
(1) python train.py -a join -r2 1 [-g2 1]
(2) python train.py -a join -r1 1 [-g1 1] -mp model_path -m tune
(3) python train.py -a join -g1 1 -g2 1 -mp model_path -m tune
```

### Addition
#### Multi-lateral Network
3 steps to build a simple multi-lateral NER architecture.
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
        encode3=self.mix_stacka('net3')
        
        # concatenate the vector
         self.encode=tf.concat([encode1,encode2,encode3],axis=-1)

        # additional process
    	...
    	
    	# 3. add a crf layer
    	self.crf_layer()
```
#### Language Correlation
This framework can also be applied to multilingual research.   
<span><img src="pic/lingual.png" width="350"> </span> <span>
<img src="pic/inter_ling.png" width="350"> </span>  

These sub-networks trained in other languages can also achieve certain performance in a new language (although not good enough), and based on this phenomenon we consider their micro F1 scores as a reflection of the correlation between languages, as shown below.  
<img src="pic/purple.png" width="500"/>  

#### Language Interaction
These sub-networks can also be combined to asynchronously train different languages simultaneously, allowing them to work together to update the model. At this point, we need to use separate output layers for each language because their sequence lengths are different.


## Suggested Datasets

+ CoNLL-2003 dataset [link](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)  
+ NYT dataset [link](https://github.com/shanzhenren/CoType)  
The augmented data set can be 3-10 times the original training data.


## Updating...

* 2017-Sep-10, Bi_NER v0.1, initial version
* 2018-Apr-05, Bi_NER v0.2, supporting easily customizing architecture and attention mechanism
* 2018-Nov-03, FlexNER v0.3, supporting different languages ( tested on English, German, Spanish, Dutch) and biomedical domain
* 2019-Mar-20, FlexNER v0.3, reconstructing the code