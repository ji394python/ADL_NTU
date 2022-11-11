python3.9 src/input_format.py $2 ./dataset/test.json

python3.9 src/run_mc.py \
  --model_name_or_path ./ckpt/mc/ \
  --do_predict \
  --cache_dir ./cache/ \
  --output_dir ./ckpt/mc/ \
  --pad_to_max_length \
  --test_file ./dataset/test.json \
  --context_file $1 \
  --output_file ./select_pred.json \
  --max_seq_length 512 

python3.9 src/run_qa.py \
  --model_name_or_path ./ckpt/qa/ \
  --do_predict \
  --cache_dir ./cache/ \
  --output_dir ./ckpt/qa/ \
  --pad_to_max_length \
  --test_file ./select_pred.json \
  --context_file $1 \
  --max_seq_length 512 \
  --doc_stride 128 

rm ./select_pred.json
python3.9 src/output_format.py $3