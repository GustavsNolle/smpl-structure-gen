import requests
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def check_url(url):
    try:
        r = requests.head(url, timeout=5)
        print(f"{url}: {r.status_code}")
    except Exception as e:
        print(f"{url}: {e}")

check_url("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz")
check_url("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hERG.csv")

m = Chem.MolFromSmiles('c1ccccc1')
sa = sascorer.calculateScore(m)
print(f"SAscore for benzene: {sa}")
