# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 src/test_slot.py --num_cnn_layers 1 --num_rnn_layers 2 --test_file "${1}" --pred_file "${2}" --ckpt_path ./ckpt/slot/best-model.pth