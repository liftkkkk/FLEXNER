# Bi_NER


Requirement:
===
    Python 2  
    Tensorflow: >1.4  


1.Usage
===
    python joint.py [-a base|att|linear|join] [-m train|restore|tune] [-g1 0|1] [-g2 0|1] [-r1 0|1] [-r2 0|1] [-mp model_path]


2.Custom
===
    def mix(self):
   
      self.base_embed=self.word_embedding_layer_base()
      
      encode1=self.mix_stacka('net1')
      encode2=self.mix_stacka('net2')
      encode3=self.mix_stacka('net3')


      if self.mask1:
        encode1=encode1*0.0
      if self.stop1:
        encode1=tf.stop_gradient(encode1)

      if self.mask2:
        encode2=encode2*0.0
      if self.stop2:
        encode2=tf.stop_gradient(encode2)
      encode=tf.concat([encode1,encode2,encode3],axis=-1)
    
.Dataset
===
CoNLL-2003 dataset in data/  
NYT dataset can be downloaded from https://github.com/shanzhenren/CoType  




Updating...
===
* 2018-Aug-26, Bi_NER v0.1, initial version
  

