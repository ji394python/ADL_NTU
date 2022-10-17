# ADL HW1

### Set up environment
```shell
pip install -r requirements.in
```

### Download cache and ckpt files
```shell
bash download.sh
```

### Predict result with testing data and submit
```shell
bash ./intent_cls.sh ./data/intent/test.json ./pred_intent.csv
bash ./slot_tag.sh ./data/slot/test.json ./pred_slot.csv
```

### How to reproduce `best_intent.ckpt`
```shell
bash preprocess.sh
bash train_intent.sh
```

### How to reproduce `best_slot.ckpt`
```shell
bash preprocess.sh
bash train_slot.sh
```