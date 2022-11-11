python3.9 src/input_format.py ./data/train.json ./dataset/train.json
python3.9 src/input_format.py ./data/valid.json ./dataset/valid.json

python3.9 src/run_mc.py \
  --do_train \
  --do_eval \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --output_dir ./ckpt/mc \
  --train_file ./dataset/train.json \
  --validation_file ./dataset/valid.json \
  --context_file ./data/context.json \
  --per_device_train_batch_size 2\
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 2 \
  --eval_accumulation_steps 2 \
  --cache_dir ./cache/ \
  --pad_to_max_length \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 

python3.9 src/run_qa.py \
  --do_train \
  --do_eval \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --output_dir ./ckpt/qa \
  --train_file ./dataset/train.json \
  --validation_file ./dataset/valid.json \
  --context_file ./data/context.json \
  --cache_dir ./cache/ \
  --per_device_train_batch_size 2\
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 2 \
  --eval_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --warmup_ratio 0.1 