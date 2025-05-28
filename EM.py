"""
EM算法模块 - 贝叶斯网络参数学习

主要功能：
1. EM算法主流程
2. E步：期望计算
3. M步：参数更新
4. 收敛检测
5. 辅助工具函数
"""

import numpy as np
import pandas as pd
import pickle
import time
from typing import Tuple, List, Dict, Any

def em_algorithm(bn, df, mis_idx, max_iter=50, tolerance=1e-4, verbose=True):
    """
    EM算法主函数
    
    Args:
        bn: 贝叶斯网络对象
        df: 观测数据DataFrame
        mis_idx: 缺失数据索引数组
        max_iter: 最大迭代次数
        tolerance: 收敛容差
        verbose: 是否显示详细信息
    
    Returns:
        tuple: (训练后的网络, 最终对数似然)
    """
    if verbose:
        print("🔄 EM算法初始化...")
        print(f"📊 参数设置: 最大迭代={max_iter}, 容差={tolerance}")
    
    log_likelihood_history = []
    log_likelihood_old = -np.inf
    
    for iteration in range(max_iter):
        if verbose:
            print(f"\n🔄 迭代 {iteration + 1}/{max_iter}")
        
        iter_start = time.time()
        
        # E步：计算期望充分统计量
        e_start = time.time()
        e_step(bn, df, mis_idx)
        e_time = time.time() - e_start
        
        # M步：更新参数
        m_start = time.time()
        m_step(bn)
        m_time = time.time() - m_start
        
        # 计算对数似然
        likelihood_start = time.time()
        log_likelihood = compute_log_likelihood(bn, df, mis_idx)
        likelihood_time = time.time() - likelihood_start
        
        log_likelihood_history.append(log_likelihood)
        
        iter_time = time.time() - iter_start
        
        if verbose:
            print(f"   ⏱️  E步: {e_time:.2f}s | M步: {m_time:.2f}s | 似然: {likelihood_time:.2f}s")
            print(f"   📈 对数似然: {log_likelihood:.6f}")
            
            if iteration > 0:
                improvement = log_likelihood - log_likelihood_old
                print(f"   📊 改进: {improvement:.8f}")
        
        # 检查收敛
        if iteration > 0 and abs(log_likelihood - log_likelihood_old) < tolerance:
            if verbose:
                print(f"✅ 算法收敛！迭代 {iteration + 1} 次")
            break
            
        log_likelihood_old = log_likelihood
    else:
        if verbose:
            print(f"⚠️  达到最大迭代次数 {max_iter}")
    
    if verbose:
        print_convergence_summary(log_likelihood_history)
    
    return bn, log_likelihood

def e_step(bn, df, mis_idx):
    """
    E步：计算期望充分统计量
    
    Args:
        bn: 贝叶斯网络
        df: 观测数据
        mis_idx: 缺失数据索引
    """
    # 重置所有节点的计数
    reset_all_counts(bn)
    
    # 处理每条记录
    for idx, (_, record) in enumerate(df.iterrows()):
        if mis_idx[idx] == -1:
            # 完整记录：直接累加计数
            process_complete_record(bn, record)
        else:
            # 缺失记录：计算期望计数
            process_missing_record(bn, record, mis_idx[idx], df.columns)

def m_step(bn):
    """
    M步：基于期望计数重新估计参数
    
    Args:
        bn: 贝叶斯网络
    """
    for node_name in bn.Pres_Graph:
        bn.normalise_cpt(node_name)

def reset_all_counts(bn):
    """重置所有节点的计数为0"""
    for node in bn.Pres_Graph.values():
        node.cpt_data['counts'] = 0.0

def process_complete_record(bn, record):
    """
    处理完整记录
    
    Args:
        bn: 贝叶斯网络
        record: 完整的观测记录
    """
    for node_name, node in bn.Pres_Graph.items():
        # 构建查询条件
        conditions = build_query_conditions(node, record)
        
        # 更新匹配行的计数
        update_node_counts(node, conditions, 1.0)

def process_missing_record(bn, record, missing_var_idx, columns):
    """
    处理有缺失数据的记录
    
    Args:
        bn: 贝叶斯网络
        record: 有缺失的观测记录
        missing_var_idx: 缺失变量索引
        columns: DataFrame列名
    """
    missing_var = columns[missing_var_idx]
    missing_node = bn.search_node(missing_var)
    
    if missing_node is None:
        return
    
    # 计算每个可能值的后验概率
    probabilities = compute_posterior_probabilities(bn, record, missing_var, missing_node)
    
    # 按概率权重更新计数
    for val, prob in enumerate(probabilities):
        if prob > 1e-10:  # 忽略极小概率
            complete_record = record.copy()
            complete_record[missing_var] = val
            
            # 为所有节点更新加权计数
            for node_name, node in bn.Pres_Graph.items():
                conditions = build_query_conditions(node, complete_record)
                update_node_counts(node, conditions, prob)

def build_query_conditions(node, record):
    """
    构建节点查询条件
    
    Args:
        node: 图节点对象
        record: 观测记录
    
    Returns:
        dict: 查询条件字典
    """
    conditions = {}
    
    # 添加父节点值
    for parent in node.Parents:
        if pd.notna(record[parent]):
            conditions[parent] = record[parent]
    
    # 添加当前节点值
    if pd.notna(record[node.Node_Name]):
        conditions[node.Node_Name] = record[node.Node_Name]
    
    return conditions

def update_node_counts(node, conditions, weight):
    """
    更新节点计数
    
    Args:
        node: 图节点对象
        conditions: 查询条件
        weight: 权重
    """
    if not conditions:  # 如果没有有效条件，跳过
        return
        
    # 构建匹配掩码
    mask = pd.Series([True] * len(node.cpt_data))
    for col, val in conditions.items():
        if col in node.cpt_data.columns:
            mask = mask & (node.cpt_data[col] == val)
    
    # 更新匹配行的计数
    if mask.any():
        node.cpt_data.loc[mask, 'counts'] += weight

def compute_posterior_probabilities(bn, record, missing_var, missing_node):
    """
    计算缺失变量的后验概率分布
    
    Args:
        bn: 贝叶斯网络
        record: 观测记录
        missing_var: 缺失变量名
        missing_node: 缺失变量节点
    
    Returns:
        list: 归一化的概率分布
    """
    probabilities = []
    
    # 计算每个可能值的条件概率
    for val in range(missing_node.nvalues):
        prob = compute_conditional_probability(missing_node, record, missing_var, val)
        probabilities.append(prob)
    
    # 归一化
    total_prob = sum(probabilities)
    if total_prob > 0:
        probabilities = [p / total_prob for p in probabilities]
    else:
        # 如果所有概率都是0，使用均匀分布
        probabilities = [1.0 / missing_node.nvalues] * missing_node.nvalues
    
    return probabilities

def compute_conditional_probability(node, record, var_name, var_value):
    """
    计算给定父节点值时变量取特定值的条件概率
    
    Args:
        node: 图节点对象
        record: 观测记录
        var_name: 变量名
        var_value: 变量值
    
    Returns:
        float: 条件概率
    """
    conditions = {}
    
    # 添加父节点条件
    for parent in node.Parents:
        if pd.notna(record[parent]):
            conditions[parent] = record[parent]
    
    # 添加当前变量值
    conditions[var_name] = var_value
    
    # 在CPT中查找概率
    mask = pd.Series([True] * len(node.cpt_data))
    for col, val in conditions.items():
        if col in node.cpt_data.columns:
            mask = mask & (node.cpt_data[col] == val)
    
    matched_rows = node.cpt_data[mask]
    if len(matched_rows) > 0 and 'p' in matched_rows.columns:
        return matched_rows['p'].iloc[0]
    else:
        return 1.0 / node.nvalues  # 均匀先验

def compute_log_likelihood(bn, df, mis_idx):
    """
    计算当前参数下的对数似然
    
    Args:
        bn: 贝叶斯网络
        df: 观测数据
        mis_idx: 缺失数据索引
    
    Returns:
        float: 对数似然值
    """
    log_likelihood = 0.0
    
    for idx, (_, record) in enumerate(df.iterrows()):
        if mis_idx[idx] == -1:  # 完整记录
            record_likelihood = compute_record_likelihood(bn, record)
            log_likelihood += np.log(max(record_likelihood, 1e-10))
        else:
            # 缺失记录的边际似然（简化处理）
            marginal_likelihood = compute_marginal_likelihood(bn, record, mis_idx[idx], df.columns)
            log_likelihood += np.log(max(marginal_likelihood, 1e-10))
    
    return log_likelihood

def compute_record_likelihood(bn, record):
    """计算完整记录的似然"""
    likelihood = 1.0
    
    for node_name, node in bn.Pres_Graph.items():
        conditions = build_query_conditions(node, record)
        
        if len(conditions) == len(node.Parents) + 1:  # 所有条件都满足
            prob = get_probability_from_cpt(node, conditions)
            likelihood *= max(prob, 1e-10)
    
    return likelihood

def compute_marginal_likelihood(bn, record, missing_var_idx, columns):
    """计算有缺失数据记录的边际似然"""
    missing_var = columns[missing_var_idx]
    missing_node = bn.search_node(missing_var)
    
    if missing_node is None:
        return 1e-5
    
    marginal_prob = 0.0
    
    # 对缺失变量的每个可能值求和
    for val in range(missing_node.nvalues):
        complete_record = record.copy()
        complete_record[missing_var] = val
        
        record_prob = compute_record_likelihood(bn, complete_record)
        marginal_prob += record_prob
    
    return marginal_prob

def get_probability_from_cpt(node, conditions):
    """从CPT中获取概率"""
    mask = pd.Series([True] * len(node.cpt_data))
    for col, val in conditions.items():
        if col in node.cpt_data.columns:
            mask = mask & (node.cpt_data[col] == val)
    
    matched_rows = node.cpt_data[mask]
    if len(matched_rows) > 0 and 'p' in matched_rows.columns:
        return matched_rows['p'].iloc[0]
    else:
        return 1.0 / node.nvalues

def print_convergence_summary(log_likelihood_history):
    """打印收敛摘要"""
    print("\n📈 收敛摘要:")
    print(f"   初始对数似然: {log_likelihood_history[0]:.6f}")
    print(f"   最终对数似然: {log_likelihood_history[-1]:.6f}")
    if len(log_likelihood_history) > 1:
        total_improvement = log_likelihood_history[-1] - log_likelihood_history[0]
        print(f"   总改进: {total_improvement:.6f}")

def save_trained_network(bn, filename):
    """
    保存训练后的网络
    
    Args:
        bn: 训练后的贝叶斯网络
        filename: 保存文件名
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(bn, f)
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def load_trained_network(filename):
    """
    加载训练后的网络
    
    Args:
        filename: 文件名
    
    Returns:
        贝叶斯网络对象
    """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None