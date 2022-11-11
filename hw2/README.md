# ADL HW2

### Set up environment
```shell
pip install -r requirements.txt
```

### How to train content-selection and question-answering
```shell
bash train.sh
```

### How to reproduce my kaggle result
```shell
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```