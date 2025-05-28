
from __future__ import division
import numpy as np
import pandas as pd
import bayesnet

# ---------------  Data helpers  -------------------------------------------- #
# def get_data(filepath: str) -> pd.DataFrame:
#     """Load records.dat & convert categorical strings to integers."""
#     '''
#     数据预处理:
#     加载: 读取 records.dat 文件（原始数据）
#     列名设置: 37个变量的标准列名
#     数据映射: 将字符串标签转换为数字编码
#     '''
#     with open(filepath, 'r', encoding='utf-8') as f:
#         raw = [line.rstrip().split() for line in f]
#     df = pd.DataFrame(raw)

#     cols = ["Hypovolemia","StrokeVolume","LVFailure","LVEDVolume","PCWP","CVP","History",
#             "MinVolSet","VentMach","Disconnect","VentTube","KinkedTube","Press","ErrLowOutput",
#             "HRBP","ErrCauter","HREKG","HRSat","BP","CO","HR","TPR","Anaphylaxis","InsuffAnesth",
#             "PAP","PulmEmbolus","FiO2","Catechol","SaO2","Shunt","PVSat","MinVol","ExpCO2",
#             "ArtCO2","VentAlv","VentLung","Intubation"]
#     df.columns = cols

#     def _map(d):
#         return {k: i for i, k in enumerate(d)}

#     mappings = {
#         **{c: {"True":0,"False":1,"?":np.nan} for c in
#            ["Hypovolemia","LVFailure","History","Disconnect","KinkedTube",
#             "ErrLowOutput","ErrCauter","Anaphylaxis","InsuffAnesth","PulmEmbolus"]},
#         **{c: {"Zero":0,"Low":1,"Normal":2,"High":3,"?":np.nan} for c in
#            ["VentMach","VentTube","Press","MinVol","ExpCO2","VentAlv","VentLung"]},
#         **{c: {"Low":0,"Normal":1,"High":2,"?":np.nan} for c in
#            ["StrokeVolume","LVEDVolume","PCWP","CVP","HRBP","HREKG",
#             "HRSat","BP","CO","HR","TPR","PAP","SaO2","PVSat","ArtCO2"]},
#         "FiO2": {"Low":0,"Normal":1,"?":np.nan},
#         "Catechol": {"Normal":0,"High":1,"?":np.nan},
#         "Shunt": {"Normal":0,"High":1,"?":np.nan},
#         "Intubation": {"Normal":0,"Esophageal":1,"OneSided":2,"?":np.nan},
#     }

#     df.replace(mappings, inplace=True)
#     return df.apply(pd.to_numeric, errors='coerce')
def get_data(filepath: str) -> pd.DataFrame:
    '''
    数据预处理: 修复版本
    加载: 读取 records.dat 文件（原始数据）
    列名设置: 37个变量的标准列名
    数据映射: 将字符串标签转换为数字编码
    '''
    print("🔍 调试 get_data():")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
    
    print(f"   文件行数: {len(lines)}")
    print(f"   第一行原始: {lines[0][:100]}...")
    
    # 修复: 先分割，然后去除双引号
    raw = []
    for line in lines:
        # 分割并去除双引号
        fields = [field.strip('"') for field in line.split()]
        raw.append(fields)
    
    df = pd.DataFrame(raw)
    print(f"   DataFrame形状: {df.shape}")
    print(f"   第一行处理后: {raw[0][:5]}")
    
    # 设置列名
    cols = ["Hypovolemia","StrokeVolume","LVFailure","LVEDVolume","PCWP","CVP","History",
            "MinVolSet","VentMach","Disconnect","VentTube","KinkedTube","Press","ErrLowOutput",
            "HRBP","ErrCauter","HREKG","HRSat","BP","CO","HR","TPR","Anaphylaxis","InsuffAnesth",
            "PAP","PulmEmbolus","FiO2","Catechol","SaO2","Shunt","PVSat","MinVol","ExpCO2",
            "ArtCO2","VentAlv","VentLung","Intubation"]
    df.columns = cols

    # 检查唯一值
    print(f"   前3列的唯一值:")
    for col in cols[:3]:
        unique_vals = df[col].unique()[:5]
        print(f"     {col}: {unique_vals}")

    # 数据映射
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
    
    # 检查映射后的情况
    print(f"   映射后前3列唯一值:")
    for col in cols[:3]:
        unique_vals = df[col].unique()[:5]
        print(f"     {col}: {unique_vals}")
    
    result = df.apply(pd.to_numeric, errors='coerce')
    
    # 检查最终结果
    nan_count = result.isna().sum().sum()
    total_values = result.size
    print(f"   NaN数量: {nan_count}/{total_values}")
    print(f"   完整记录数: {(~result.isna().any(axis=1)).sum()}")
    
    return result

# ---------------  Network initialisation  ---------------------------------- #
def setup_network(bif_alarm: str, dat_records: str):
    '''
    功能: 一站式建立完整的贝叶斯网络系统
        读取网络结构: 从 .bif 文件构建网络
        设置马尔可夫毯: 预计算所有节点的马尔可夫毯
        加载数据: 处理观测数据
        参数初始化: 为EM算法准备初始参数
        缺失数据索引: 记录每行缺失变量的位置
    '''
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
    '''
    #缺失数据定位
    ## 返回每行第一个缺失变量的索引，完整行返回-1
    '''
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
