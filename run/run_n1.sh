model_dir={model-dir}

if [ ! -e $model_dir ] ; then
	echo 'create',$model_dir
    mkdir -p $model_dir
elif [ ! -d $model_dir ] ; then
    echo "$dir already exists but is not a directory" 1>&2
else 
	echo $model_dir 'exists'
fi


CUDA_VISIBLE_DEVICES=0 python3 run_n1.py \
--use_random_embed 0 \
--model_path {Your checkpoint} \
--algorithm Joint \
--mode train \
--lang zh \
--corpus new \
--train_h5 {training-data.h5} \
--test_h5 {test-data.h5} \
--test_pkl {test-data.pkl} \
--results_report ../report.txt \
--gradient_stop_net1 0 \
--mask_net1 0 \
--gradient_stop_net2 1 \
--mask_net2 1 \
--save_model_dir $model_dir \
--predict_file output.txt \
--word_embed_h5 {word embeddings} \
--word_embed_voc {vocab}} \
--char_voc {char-vocab} \
--build_voc {optional: build-vocab} \
--save_step 100 
