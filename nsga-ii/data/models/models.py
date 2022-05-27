
import pandas as pd
import joblib
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

def train_model(chid, n_estimators=100, n_jobs=8):
    assay_file = f'./processed/{chid}.csv'
    df = pd.read_csv(assay_file)
    fingerprints =  np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024)) for smiles in df.smiles])
    target = np.array(df.label)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs).fit(fingerprints, target)
    joblib.dump(clf, f"./clf_{chid}.joblib")
    return None

def process_chembl_data(chid):
    df = pd.read_csv(f'./raw/{chid}.csv', sep=';')
    df['pChEMBL Value'] = df['pChEMBL Value'].fillna(0)
    df = df[['Smiles', 'pChEMBL Value']].dropna()
    df.columns = ['smiles', 'value']
    label = np.array([1 if x > 8 else 0 for x in df.value])
    df['label'] = label
    df.to_csv(f'./processed/{chid}.csv', index=None)
    return None 

# process_chembl_data("DAPK1")
# process_chembl_data("DRP1")
# process_chembl_data("ZIPK")

# process_chembl_data("HERG")
# process_chembl_data("SCN2A")

# process_chembl_data("D2")
# process_chembl_data("HT2A")
# process_chembl_data("HT2B")

process_chembl_data("ADRA1")
process_chembl_data("HRH3")

# train_model("DAPK1")
# train_model("DRP1")
# train_model("ZIPK")

# train_model("HERG")
# train_model("SCN2A")

# train_model("D2")
# train_model("HT2A")
# train_model("HT2B")

train_model("ADRA1")
train_model("HRH3")
