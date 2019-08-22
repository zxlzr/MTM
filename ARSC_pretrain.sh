export GLUE_DIR="data/Amazon_few_shot"

python3 ARSC_lm_finetuning.py \
--train_corpus data/Amazon_corpus.txt \
--bert_model bert-base-uncased-file \
--do_lower_case \
--output_dir /tmp/finetuned_lm/ \
--train_batch_size 32 \
--do_train

cp /tmp/Amazon_maml_output1/config.json /tmp/finetuned_lm/ 
cp /tmp/Amazon_maml_output1/vocab.txt /tmp/finetuned_lm/ 

python3 run_classifier_maml.py \
  --task_name Amazon \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR \
  --bert_model /tmp/finetuned_lm/ \
  --max_seq_length 128 \
  --inner_learning_rate 2e-6 \
  --outer_learning_rate 1e-5 \
  --output_dir /tmp/Amazon_maml_output5/