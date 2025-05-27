
from __future__ import division
import numpy as np
import pandas as pd
import bayesnet

# ---------------  Data helpers  -------------------------------------------- #
def get_data(filepath: str) -> pd.DataFrame:
    """Load records.dat & convert categorical strings to integers."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = [line.rstrip().split() for line in f]
    df = pd.DataFrame(raw)

    cols = ["Hypovolemia","StrokeVolume","LVFailure","LVEDVolume","PCWP","CVP","History",
            "MinVolSet","VentMach","Disconnect","VentTube","KinkedTube","Press","ErrLowOutput",
            "HRBP","ErrCauter","HREKG","HRSat","BP","CO","HR","TPR","Anaphylaxis","InsuffAnesth",
            "PAP","PulmEmbolus","FiO2","Catechol","SaO2","Shunt","PVSat","MinVol","ExpCO2",
            "ArtCO2","VentAlv","VentLung","Intubation"]
    df.columns = cols

    def _map(d):
        return {k: i for i, k in enumerate(d)}

    mappings = {
        **{c: {"True":0,"False":1,"?":np.nan} for c in
           ["Hypovolemia","LVFailure","History","Disconnect","KinkedTube",
            "ErrLowOutput","ErrCauter","Anaphylaxis","InsuffAnesth","PulmEmbolus"]},
        **{c: {"Zero":0,"Low":1,"Normal":2,"High":3,"?":np.nan} for c in
           ["VentMach","VentTube","Press","MinVol","ExpCO2","VentAlv","VentLung"]},
        **{c: {"Low":0,"Normal":1,"High":2,"?":np.nan} for c in
           ["StrokeVolume","LVEDVolume","PCWP","CVP","HRBP","HREKG",
            "HRSat","BP","CO","HR","TPR","PAP","SaO2","PVSat","ArtCO2"]},
        "FiO2": {"Low":0,"Normal":1,"?":np.nan},
        "Catechol": {"Normal":0,"High":1,"?":np.nan},
        "Shunt": {"Normal":0,"High":1,"?":np.nan},
        "Intubation": {"Normal":0,"Esophageal":1,"OneSided":2,"?":np.nan},
    }

    df.replace(mappings, inplace=True)
    return df.apply(pd.to_numeric, errors='coerce')

# ---------------  Network initialisation  ---------------------------------- #
def setup_network(bif_alarm: str, dat_records: str):
    print("0: Reading Network . . . ")
    Alarm = bayesnet.read_network(bif_alarm)

    print("1: Setting Markov Blankets . . . ")
    Alarm.set_mb()

    print("2: Getting data from records . . . ")
    df = get_data(dat_records)

    print("3: Initialising parameters . . . ")
    init_params(df, Alarm)

    print("4: Getting missing data indexes . . . ")
    mis_index = get_missing_index(df)

    return Alarm, df, mis_index

# ---------------  EM initialisation helpers  ------------------------------- #
def get_missing_index(df):
    idx = []
    for _, row in df.iterrows():
        if row.isnull().any():
            idx.append(int(np.where(row.isna())[0][0]))
        else:
            idx.append(-1)
    return idx

def init_params(df, net):
    for node in net.Pres_Graph.values():
        # simple uniform prior until EM runs
        node.cpt_data["counts"] = 1.0
        net.normalise_cpt(node.Node_Name)
