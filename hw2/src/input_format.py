import json
import os
import sys

if __name__ =="__main__":
    
    with open(sys.argv[1], 'r',encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
    json.dump({'data': data}, open(sys.argv[2], 'w',encoding='utf-8'), indent=2, ensure_ascii=False)
