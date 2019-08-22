
# python3 run_classifier.py \
#   --task_name Amazon \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir $GLUE_DIR \
#   --bert_model bert-base-uncased-file \
#   --max_seq_length 128 \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2.0 \
  # --output_dir /tmp/Amazon_output/
# python3 run_classifier_maml.py \
#   --task_name Amazon \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir $GLUE_DIR \
#   --bert_model /tmp/finetuned_lm/ \
#   --max_seq_length 128 \
#   --inner_learning_rate 2e-6 \
#   --outer_learning_rate 1e-5 \
#   --output_dir /tmp/Amazon_maml_output5/
export GLUE_DIR="data/Amazon_few_shot"

# python3 run_classifier_maml.py \
#   --task_name Amazon \
#   --is_init True \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir $GLUE_DIR \
#   --bert_model bert-base-uncased-file \
#   --max_seq_length 128 \
#   --inner_learning_rate 2e-6 \
#   --outer_learning_rate 1e-5 \
#   --output_dir /tmp/Amazon_maml_no_pretrain/

python3 run_classifier_maml.py \
  --task_name Amazon \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR \
  --bert_model /tmp/Amazon_fomaml_with_pretrain/ \
  --max_seq_length 128 \
  --inner_learning_rate 2e-6 \
  --outer_learning_rate 1e-5 \
  --output_dir /tmp/Amazon_fomaml_with_pretrain1/