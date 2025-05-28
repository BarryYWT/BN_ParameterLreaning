
from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd
import itertools
import re 
###############################################################################
# Bayesian Network core classes + utilities
# This version strips the quotes found in the Alarm .bif file so that internal
# node names match those used in records.dat. It’s been modernised for pandas ≥ 2.
###############################################################################

# ----------------------------  Graph Node  ---------------------------------- #
class Graph_Node:
    '''
        提取gold_alarm.bif里存的单个节点的----BN结构和CPT概率值
        
        解析 .bif 文件
        为每个变量创建 Graph_Node 对象  
        提取父子关系建立网络结构
        存储概率表数据
        通过 normalise_cpt() 将原始数据转换为便于计算的DataFrame格式
    '''
    def __init__(self, name: str, n: int, vals):
        self.Node_Name  = name            # variable name
        self.nvalues    = n               # number of categorical levels
        self.values     = vals            # list of level labels
        self.Children   = []              # indices (ints) of child nodes
        self.Parents    = []              # names (str) of parent nodes

        # CPT is stored twice:
        #   * raw list in self.CPT (legacy)
        #   * tidy pandas.DataFrame in self.cpt_data (preferred)
        self.CPT        = []
        self.cpt_data   = pd.DataFrame()

    # Convenience:
    def add_child(self, child_idx: int):
        if child_idx not in self.Children:
            self.Children.append(child_idx)

    def set_Parents(self, parents):
        self.Parents = parents

    def set_CPT(self, values):
        self.CPT = list(values)

    def set_cpt_data(self, df: pd.DataFrame):
        self.cpt_data = df.copy()


# ----------------------------  Network  ------------------------------------- #
class network:
    """A Bayesian network made of Graph_Node objects
    把从 .bif 文件解析出的所有 Graph_Node 对象组织成一个完整的贝叶斯网络。
    """

    def __init__(self):
        self.Pres_Graph = OrderedDict()   # name → Graph_Node
        self.MB         = OrderedDict()   # name → list of blanket node names

    # -- Node look‑ups ------------------------------------------------------- #
    def addNode(self, node: Graph_Node):
        self.Pres_Graph[node.Node_Name] = node

    def search_node(self, name: str):
        return self.Pres_Graph.get(name)

    def get_index(self, name: str):
        try:
            return list(self.Pres_Graph.keys()).index(name)
        except ValueError:
            print(f'No node of the name: {name}')
            return None

    # -- Helpers ------------------------------------------------------------- #
    def get_parent_nodes(self, node: Graph_Node):
        return [self.search_node(p) for p in node.Parents]
        '''
        功能: 获取指定节点的所有父节点对象
            输入: 一个 Graph_Node 对象
            输出: 父节点对象的列表
            实现: 遍历节点的父节点名称列表，通过 search_node() 查找对应的节点对象
        '''
    # def normalise_cpt(self, X: str):
    #     """
    #             ## `normalise_cpt(self, X: str)`
    #             **功能**: 将节点的条件概率表(CPT)标准化为概率分布

    #             ### 两个主要步骤：

    #             #### 1. **格式转换** (如果 DataFrame 为空)
    #             - 验证父节点存在性
    #             - 生成所有可能的父节点配置组合
    #             - 将原始 CPT 列表转换为结构化的 pandas DataFrame
    #             - 每行包含: `父节点值 + 子节点值 + counts值`
    #     """
    #     node    = self.Pres_Graph[X]
    #     parents = node.Parents           # 父节点名字列表
    #     nvals   = node.nvalues           # X 自己的取值个数

    #     # ---------- 1.  若 DataFrame 为空，就把旧版 list CPT 转成 DF ----------
    #     if node.cpt_data.empty:
    #         # 1-a  确认父节点都已被成功解析，否则立刻报错给调用者
    #         missing = [p for p in parents if self.search_node(p) is None]
    #         if missing:
    #             raise ValueError(f'节点 “{X}” 的父节点不存在于网络中: {missing}')

    #         # 1-b  生成每个父节点的取值范围
    #         parent_cards = [self.search_node(p).nvalues for p in parents]

    #         # 1-c  用 itertools.product 得到所有 (父配置, X取值) 的笛卡儿积
    #         grid = list(itertools.product(*[range(k) for k in parent_cards], range(nvals)))

    #         # 1-d  CPT 长度必须与配置数量一致
    #         if len(grid) != len(node.CPT):
    #             raise ValueError(
    #                 f'CPT 长度 {len(node.CPT)} 与期望配置数 {len(grid)} 不符，请检查 BIF 文件'
    #             )

    #         # 1-e  组装成 DataFrame
    #         rows = []
    #         for cfg, prob in zip(grid, node.CPT):
    #             row = {p: cfg[i] for i, p in enumerate(parents)}  # 父变量列
    #             row[X]  = cfg[-1]                                 # 子变量列
    #             row['counts'] = float(prob)                       # 计数列
    #             rows.append(row)
    #         cpt = pd.DataFrame(rows)
    #     else:
    #         cpt = node.cpt_data.copy()   # 已经是 DF，直接复制一份

    #     # ---------- 2.  对每一种父配置做行归一化 ----------
    #     def _norm(group: pd.DataFrame) -> pd.DataFrame:
    #         counts = group['counts'].astype(float).to_numpy()
    #         counts[counts == 0] = 5e-6           # 拉普拉斯平滑，防 0
    #         group = group.assign(p = counts / counts.sum())
    #         return group
    #     '''
    #         #### 2. **概率归一化**
    #         ### 关键特性：
    #         - **按父配置分组**: 对每种父节点配置单独归一化(确保每种父配置下子节点概率和为1)
    #         - **拉普拉斯平滑**: 避免零概率问题
    #         - **双存储**: 同时保持原始CPT和DataFrame格式

    #         ### 举例说明：
    #         对于节点 "StrokeVolume" (父节点: LVFailure, Hypovolemia)：
    #         - 原始CPT: `[0.98, 0.5, 0.95, 0.05, ...]` 
    #         - 转换后每组归一化为概率分布，确保每种父配置下子节点概率和为1

    #         这是贝叶斯网络参数学习的基础数据预处理步骤。
    #     '''
    #     node.cpt_data = cpt.groupby(parents, group_keys=False).apply(_norm)
        
    def normalise_cpt(self, X: str):
        #"""将节点的条件概率表(CPT)标准化为概率分布"""
        node    = self.Pres_Graph[X]
        parents = node.Parents           # 父节点名字列表
        nvals   = node.nvalues           # X 自己的取值个数

        # ---------- 1.  若 DataFrame 为空，就把旧版 list CPT 转成 DF ----------
        """
                ## `normalise_cpt(self, X: str)`
                **功能**: 将节点的条件概率表(CPT)标准化为概率分布

                ### 两个主要步骤：

                #### 1. **格式转换** (如果 DataFrame 为空)
                - 验证父节点存在性
                - 生成所有可能的父节点配置组合
                - 将原始 CPT 列表转换为结构化的 pandas DataFrame
                - 每行包含: `父节点值 + 子节点值 + counts值`
        """
        if node.cpt_data.empty:
            # 1-a  确认父节点都已被成功解析，否则立刻报错给调用者
            missing = [p for p in parents if self.search_node(p) is None]
            if missing:
                raise ValueError(f'节点 "{X}" 的父节点不存在于网络中: {missing}')

            # 1-b  生成每个父节点的取值范围
            parent_cards = [self.search_node(p).nvalues for p in parents]

            # 1-c  用 itertools.product 得到所有 (父配置, X取值) 的笛卡儿积
            grid = list(itertools.product(*[range(k) for k in parent_cards], range(nvals)))

            # 1-d  CPT 长度必须与配置数量一致
            if len(grid) != len(node.CPT):
                raise ValueError(
                    f'CPT 长度 {len(node.CPT)} 与期望配置数 {len(grid)} 不符，请检查 BIF 文件'
                )

            # 1-e  组装成 DataFrame
            rows = []
            for cfg, prob in zip(grid, node.CPT):
                row = {}
                # 添加父节点值
                for i, parent_name in enumerate(parents):
                    row[parent_name] = cfg[i]
                # 添加子节点值和概率
                row[X] = cfg[-1]  # 最后一个是子节点的值
                row['counts'] = prob
                rows.append(row)
            
            cpt = pd.DataFrame(rows)
        else:
            cpt = node.cpt_data.copy()   # 已经是 DF，直接复制一份

        # ---------- 2.  对每一种父配置做行归一化 ----------
        '''
            #### 2. **概率归一化**
            ### 关键特性：
            - **按父配置分组**: 对每种父节点配置单独归一化(确保每种父配置下子节点概率和为1)
            - **拉普拉斯平滑**: 避免零概率问题
            - **双存储**: 同时保持原始CPT和DataFrame格式

            ### 举例说明：
            对于节点 "StrokeVolume" (父节点: LVFailure, Hypovolemia)：
            - 原始CPT: `[0.98, 0.5, 0.95, 0.05, ...]` 
            - 转换后每组归一化为概率分布，确保每种父配置下子节点概率和为1

            这是贝叶斯网络参数学习的基础数据预处理步骤。
        '''
        def _norm(group: pd.DataFrame) -> pd.DataFrame:
            counts = group['counts'].astype(float).to_numpy()
            counts[counts == 0] = 5e-6           # 拉普拉斯平滑，防 0
            group = group.assign(p = counts / counts.sum())
            return group

        # 处理没有父节点的情况
        if not parents:
            # 没有父节点，直接对整个表做归一化
            node.cpt_data = _norm(cpt)
        else:
            # 有父节点，按父配置分组归一化
            node.cpt_data = cpt.groupby(parents, group_keys=False).apply(_norm)
        
        
        
        
        
    # -- Markov blankets ----------------------------------------------------- #
    def set_mb(self):
        '''
        #批量预计算并存储所有节点的马尔可夫毯
        '''
        for name in self.Pres_Graph:
            self.MB[name] = markov_blanket(self, name)

def read_network(bif_filepath):
    """
    读取 .bif 文件，构建 network 对象（包含 CPT）。
    采用正则解析，避免把注释 / 括号 / 分号误当成父节点或数字。
    
    层级关系：
    .bif 文件
        ↓ (read_network 解析)
    network 对象
        ↓ (包含多个)
    Graph_Node 对象们
    
    read_network() - 工厂函数
    作用: 构建网络对象的工厂函数
    输入: .bif 文件路径
    输出: 完整的 network 对象
    类比: 就像一个"生产线"
    
    class network - 整个网络容器
    作用: 管理整个贝叶斯网络
    存储: 所有 Graph_Node 对象的集合
    功能: 提供节点查找、概率计算、网络操作等方法
    类比: 就像一个"组装车间"
    
    class Graph_Node - 单个节点
    作用: 表示贝叶斯网络中的一个节点
    存储: 单个变量的信息（名称、取值、父子关系、概率表）
    类比: 就像一个"零件"
    """
    net = network()

    with open(bif_filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:             # EOF
                break
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            tokens = line.split()
            first_word = tokens[0]

            # ───────────────────── variable 块 ─────────────────────
            if first_word == "variable":
                # 正则抓引号里的变量名
                var_name = re.search(r'"([^"]+)"', line).group(1)
                # 下一行一定包含数值枚举
                enum_line = f.readline()
                values = re.findall(r'"([^"]+)"', enum_line)
                node = Graph_Node(name=var_name, n=len(values), vals=values)
                net.addNode(node)
                continue

            # ──────────────────── probability 块 ───────────────────
            if first_word == "probability":
                header = line
                # 保证 header 一直读到 “)” 结束
                while ")" not in header:
                    header += f.readline()

                # 1⃣ 把括号里的所有变量名取出来
                var_list = re.findall(r'"([^"]+)"', header)
                child_name, *parents = var_list

                # 2⃣ 建立父子关系
                child_node = net.search_node(child_name)
                if child_node is None:
                    raise ValueError(f'variable "{child_name}" 尚未在文件前面声明')
                child_idx = net.get_index(child_name)
                child_node.set_Parents(parents)

                for p in parents:
                    par_node = net.search_node(p)
                    if par_node is None:
                        raise ValueError(f'父节点 "{p}" 未声明 (检查 variable 次序)')
                    par_node.add_child(child_idx)

                # 3⃣ 读取 table 数字直到遇到分号 ;
                nums = []
                while True:
                    row = f.readline()
                    if not row:
                        raise EOFError("BIF 在 table 定义处意外结束")
                    nums.extend([float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', row)])
                    if ";" in row:
                        break
                child_node.CPT = nums
        # end-while file
    # 4⃣ 用 CPT 列表转换为 DataFrame 并做概率归一化
    for var in net.Pres_Graph.keys():
        net.normalise_cpt(var)

    # 设置 Markov blanket（可选）
    net.set_mb()
    return net

# -----------------------------  Utilities  ---------------------------------- #
def markov_blanket(net: network, var: str):
    '''
        输入:
        net: 贝叶斯网络对象
        var: 节点名称（字符串）
        输出:
        该节点的马尔可夫毯节点名称列表
    '''
    node = net.search_node(var)
    mb   = set()

    # parents
    mb.update(node.Parents)
    # children
    for idx in node.Children:
        child_name = list(net.Pres_Graph.keys())[idx]
        mb.add(child_name)
        # spouses
        mb.update(net.search_node(child_name).Parents)

    mb.discard(var)
    return list(mb)

# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('Run main.py instead of bayesnet.py')
