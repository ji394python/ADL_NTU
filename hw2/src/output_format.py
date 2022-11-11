import pandas as pd 
import json
from pathlib import Path
import sys
if __name__ == '__main__':
    df = json.loads(Path('./ckpt/qa/test_predictions.json').read_text())
    idx,answers = [],[]
    for k,v in df.items():
        idx.append(str(k))
        answers.append(str(v))
        
    df2 = pd.DataFrame({'id':idx,'answer':answers})
    df2.to_csv(sys.argv[1],index=False,encoding='utf-8')