python3.9 src/down_pre.py \
    --cache_dir ./cache/ \
    --model_name_or_path hfl/chinese-roberta-wwm-ext 
    
wget https://www.dropbox.com/s/xigx7f5ylnbjxz4/pytorch_model.bin?dl=0 -O ./ckpt/mc/pytorch_model.bin
wget https://www.dropbox.com/s/pt3h6gk58cit9o6/pytorch_model.bin?dl=0 -O ./ckpt/qa/pytorch_model.bin