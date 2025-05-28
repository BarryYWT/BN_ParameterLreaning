"""
EM算法模块 - 贝叶斯网络参数学习 (调试版本)

修复了对数似然计算问题，添加了详细的调试信息
"""

import numpy as np
import pandas as pd
import pickle
import time
from typing import Tuple, List, Dict, Any

def em_algorithm(bn, df, mis_idx, max_iter=50, tolerance=1e-4, verbose=True):
    """
    EM算法主函数 - 调试版本
    """
    if verbose:
        print("🔄 EM算法初始化...")
        print(f"📊 参数设置: 最大迭代={max_iter}, 容差={tolerance}")
        
        # 🔍 调试信息：数据基本情况
        print("\n🔍 调试信息:")
        print(f"   数据形状: {df.shape}")
        print(f"   网络节点数: {len(bn.Pres_Graph)}")
        print(f"   完整记录数: {sum(1 for x in mis_idx if x == -1)}")
        print(f"   缺失记录数: {sum(1 for x in mis_idx if x != -1)}")
        
        # 检查数据样本
        print(f"\n🔍 数据样本:")
        print(df.head(3))
        print(f"\n🔍 数据类型:")
        print(df.dtypes.head(5))
        
        # 检查第一个节点的CPT
        first_node_name = list(bn.Pres_Graph.keys())[0]
        first_node = bn.Pres_Graph[first_node_name]
        print(f"\n🔍 节点 '{first_node_name}' CPT样本:")
        print(first_node.cpt_data.head())
        print(f"   CPT形状: {first_node.cpt_data.shape}")
        print(f"   是否有'p'列: {'p' in first_node.cpt_data.columns}")
        
    log_likelihood_history = []
    log_likelihood_old = -np.inf
    
    for iteration in range(max_iter):
        if verbose:
            print(f"\n🔄 迭代 {iteration + 1}/{max_iter}")
        
        iter_start = time.time()
        
        # E步：计算期望充分统计量
        e_start = time.time()
        e_step_debug(bn, df, mis_idx, verbose and iteration == 0)
        e_time = time.time() - e_start
        
        # M步：更新参数
        m_start = time.time()
        m_step_debug(bn, verbose and iteration == 0)
        m_time = time.time() - m_start
        
        # 计算对数似然
        likelihood_start = time.time()
        log_likelihood = compute_log_likelihood_debug(bn, df, mis_idx, verbose and iteration == 0)
        likelihood_time = time.time() - likelihood_start
        
        log_likelihood_history.append(log_likelihood)
        
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

def e_step_debug(bn, df, mis_idx, debug=False):
    """E步 - 调试版本"""
    if debug:
        print("\n🔍 E步调试:")
    
    # 重置所有节点的计数
    reset_all_counts(bn)
    
    complete_count = 0
    missing_count = 0
    
    # 处理每条记录
    for idx, (_, record) in enumerate(df.iterrows()):
        if mis_idx[idx] == -1:
            # 完整记录：直接累加计数
            process_complete_record_debug(bn, record, debug and idx < 3)
            complete_count += 1
        else:
            # 缺失记录：计算期望计数
            process_missing_record_debug(bn, record, mis_idx[idx], df.columns, debug and idx < 3)
            missing_count += 1
    
    if debug:
        print(f"   处理完整记录: {complete_count}")
        print(f"   处理缺失记录: {missing_count}")
        
        # 检查计数更新情况
        first_node = list(bn.Pres_Graph.values())[0]
        total_counts = first_node.cpt_data['counts'].sum()
        print(f"   第一个节点总计数: {total_counts}")

def m_step_debug(bn, debug=False):
    """M步 - 调试版本"""
    if debug:
        print("\n🔍 M步调试:")
        
        # 检查归一化前的计数
        first_node = list(bn.Pres_Graph.values())[0]
        print(f"   归一化前计数范围: {first_node.cpt_data['counts'].min():.4f} - {first_node.cpt_data['counts'].max():.4f}")
    
    for node_name in bn.Pres_Graph:
        bn.normalise_cpt(node_name)
    
    if debug:
        # 检查归一化后的概率
        first_node = list(bn.Pres_Graph.values())[0]
        if 'p' in first_node.cpt_data.columns:
            print(f"   归一化后概率范围: {first_node.cpt_data['p'].min():.4f} - {first_node.cpt_data['p'].max():.4f}")
            print(f"   概率总和检查: {first_node.cpt_data['p'].sum():.4f}")

def process_complete_record_debug(bn, record, debug=False):
    """处理完整记录 - 调试版本"""
    if debug:
        print(f"\n🔍 处理完整记录调试:")
        print(f"   记录内容: {dict(record.head())}")
    
    updates_made = 0
    
    for node_name, node in bn.Pres_Graph.items():
        # 构建查询条件
        conditions = build_query_conditions_debug(node, record, debug and updates_made < 2)
        
        # 更新匹配行的计数
        updated = update_node_counts_debug(node, conditions, 1.0, debug and updates_made < 2)
        if updated:
            updates_made += 1
    
    if debug:
        print(f"   成功更新节点数: {updates_made}")

def build_query_conditions_debug(node, record, debug=False):
    """构建查询条件 - 调试版本"""
    conditions = {}
    
    # 添加父节点值
    for parent in node.Parents:
        if pd.notna(record[parent]):
            conditions[parent] = record[parent]
    
    # 添加当前节点值
    if pd.notna(record[node.Node_Name]):
        conditions[node.Node_Name] = record[node.Node_Name]
    
    if debug:
        print(f"     节点 {node.Node_Name}: 条件 = {conditions}")
    
    return conditions

def update_node_counts_debug(node, conditions, weight, debug=False):
    """更新节点计数 - 调试版本"""
    if not conditions:
        if debug:
            print(f"     跳过: 无有效条件")
        return False
    
    # 构建匹配掩码
    mask = pd.Series([True] * len(node.cpt_data))
    for col, val in conditions.items():
        if col in node.cpt_data.columns:
            mask = mask & (node.cpt_data[col] == val)
        else:
            if debug:
                print(f"     警告: 列 '{col}' 不在CPT中")
            return False
    
    # 更新匹配行的计数
    matched_count = mask.sum()
    if matched_count > 0:
        old_count = node.cpt_data.loc[mask, 'counts'].iloc[0] if matched_count > 0 else 0
        node.cpt_data.loc[mask, 'counts'] += weight
        new_count = node.cpt_data.loc[mask, 'counts'].iloc[0] if matched_count > 0 else 0
        
        if debug:
            print(f"     匹配行数: {matched_count}, 计数: {old_count:.2f} -> {new_count:.2f}")
        return True
    else:
        if debug:
            print(f"     警告: 无匹配行")
        return False

def process_missing_record_debug(bn, record, missing_var_idx, columns, debug=False):
    """处理缺失记录 - 调试版本"""
    missing_var = columns[missing_var_idx]
    missing_node = bn.search_node(missing_var)
    
    if missing_node is None:
        if debug:
            print(f"     警告: 找不到缺失变量节点 '{missing_var}'")
        return
    
    if debug:
        print(f"\n🔍 处理缺失记录: 变量 '{missing_var}'")
    
    # 计算每个可能值的后验概率
    probabilities = compute_posterior_probabilities_debug(bn, record, missing_var, missing_node, debug)
    
    if debug:
        print(f"   后验概率: {probabilities}")
    
    # 按概率权重更新计数
    for val, prob in enumerate(probabilities):
        if prob > 1e-10:
            complete_record = record.copy()
            complete_record[missing_var] = val
            
            # 为所有节点更新加权计数
            for node_name, node in bn.Pres_Graph.items():
                conditions = build_query_conditions_debug(node, complete_record, False)
                update_node_counts_debug(node, conditions, prob, False)

def compute_posterior_probabilities_debug(bn, record, missing_var, missing_node, debug=False):
    """计算后验概率 - 调试版本"""
    probabilities = []
    
    for val in range(missing_node.nvalues):
        prob = compute_conditional_probability_debug(missing_node, record, missing_var, val, debug and val < 2)
        probabilities.append(prob)
    
    # 归一化
    total_prob = sum(probabilities)
    if total_prob > 0:
        probabilities = [p / total_prob for p in probabilities]
    else:
        probabilities = [1.0 / missing_node.nvalues] * missing_node.nvalues
        if debug:
            print(f"   警告: 所有概率为0，使用均匀分布")
    
    return probabilities

def compute_conditional_probability_debug(node, record, var_name, var_value, debug=False):
    """计算条件概率 - 调试版本"""
    conditions = {}
    
    # 添加父节点条件
    for parent in node.Parents:
        if pd.notna(record[parent]):
            conditions[parent] = record[parent]
    
    # 添加当前变量值
    conditions[var_name] = var_value
    
    if debug:
        print(f"     查询条件: {conditions}")
    
    # 在CPT中查找概率
    mask = pd.Series([True] * len(node.cpt_data))
    for col, val in conditions.items():
        if col in node.cpt_data.columns:
            mask = mask & (node.cpt_data[col] == val)
    
    matched_rows = node.cpt_data[mask]
    if len(matched_rows) > 0 and 'p' in matched_rows.columns:
        prob = matched_rows['p'].iloc[0]
        if debug:
            print(f"     找到概率: {prob:.6f}")
        return prob
    else:
        prob = 1.0 / node.nvalues
        if debug:
            print(f"     未找到，使用均匀先验: {prob:.6f}")
        return prob

def compute_log_likelihood_debug(bn, df, mis_idx, debug=False):
    """计算对数似然 - 调试版本"""
    if debug:
        print("\n🔍 对数似然计算调试:")
    
    log_likelihood = 0.0
    complete_processed = 0
    missing_processed = 0
    zero_likelihood_count = 0
    
    for idx, (_, record) in enumerate(df.iterrows()):
        if mis_idx[idx] == -1:  # 完整记录
            record_likelihood = compute_record_likelihood_debug(bn, record, debug and complete_processed < 3)
            
            if record_likelihood > 0:
                log_likelihood += np.log(record_likelihood)
            else:
                zero_likelihood_count += 1
                log_likelihood += np.log(1e-10)  # 避免log(0)
            
            complete_processed += 1
            
        else:
            # 缺失记录的边际似然
            marginal_likelihood = compute_marginal_likelihood_debug(bn, record, mis_idx[idx], df.columns, debug and missing_processed < 2)
            
            if marginal_likelihood > 0:
                log_likelihood += np.log(marginal_likelihood)
            else:
                log_likelihood += np.log(1e-10)
            
            missing_processed += 1
    
    if debug:
        print(f"   处理完整记录: {complete_processed}")
        print(f"   处理缺失记录: {missing_processed}")
        print(f"   零似然记录数: {zero_likelihood_count}")
        print(f"   对数似然总和: {log_likelihood:.6f}")
        
        if zero_likelihood_count > 0:
            print(f"   ⚠️  有 {zero_likelihood_count} 条记录似然为0！")
    
    return log_likelihood

def compute_record_likelihood_debug(bn, record, debug=False):
    """计算记录似然 - 调试版本"""
    likelihood = 1.0
    zero_prob_nodes = []
    
    if debug:
        print(f"\n🔍 计算记录似然:")
    
    for node_name, node in bn.Pres_Graph.items():
        conditions = build_query_conditions_debug(node, record, False)
        
        if len(conditions) == len(node.Parents) + 1:  # 所有条件都满足
            prob = get_probability_from_cpt_debug(node, conditions, debug and len(zero_prob_nodes) < 3)
            if prob > 0:
                likelihood *= prob
            else:
                zero_prob_nodes.append(node_name)
                likelihood *= 1e-10
    
    if debug:
        print(f"   最终似然: {likelihood:.10f}")
        if zero_prob_nodes:
            print(f"   零概率节点: {zero_prob_nodes[:3]}")
    
    return likelihood

def get_probability_from_cpt_debug(node, conditions, debug=False):
    """从CPT获取概率 - 调试版本"""
    mask = pd.Series([True] * len(node.cpt_data))
    for col, val in conditions.items():
        if col in node.cpt_data.columns:
            mask = mask & (node.cpt_data[col] == val)
    
    matched_rows = node.cpt_data[mask]
    if len(matched_rows) > 0 and 'p' in matched_rows.columns:
        prob = matched_rows['p'].iloc[0]
        if debug:
            print(f"   节点 {node.Node_Name}: 条件={conditions}, 概率={prob:.6f}")
        return prob
    else:
        prob = 1.0 / node.nvalues
        if debug:
            print(f"   节点 {node.Node_Name}: 未找到匹配，使用 {prob:.6f}")
        return prob

def compute_marginal_likelihood_debug(bn, record, missing_var_idx, columns, debug=False):
    """计算边际似然 - 调试版本"""
    missing_var = columns[missing_var_idx]
    missing_node = bn.search_node(missing_var)
    
    if missing_node is None:
        return 1e-5
    
    marginal_prob = 0.0
    
    if debug:
        print(f"\n🔍 计算边际似然: 变量 '{missing_var}'")
    
    # 对缺失变量的每个可能值求和
    for val in range(missing_node.nvalues):
        complete_record = record.copy()
        complete_record[missing_var] = val
        
        record_prob = compute_record_likelihood_debug(bn, complete_record, False)
        marginal_prob += record_prob
        
        if debug and val < 2:
            print(f"   值 {val}: 概率 = {record_prob:.8f}")
    
    if debug:
        print(f"   边际概率: {marginal_prob:.8f}")
    
    return marginal_prob

def reset_all_counts(bn):
    """重置所有节点的计数为0"""
    for node in bn.Pres_Graph.values():
        node.cpt_data['counts'] = 0.0

def print_convergence_summary(log_likelihood_history):
    """打印收敛摘要"""
    print("\n📈 收敛摘要:")
    print(f"   初始对数似然: {log_likelihood_history[0]:.6f}")
    print(f"   最终对数似然: {log_likelihood_history[-1]:.6f}")
    if len(log_likelihood_history) > 1:
        total_improvement = log_likelihood_history[-1] - log_likelihood_history[0]
        print(f"   总改进: {total_improvement:.6f}")

def save_trained_network(bn, filename):
    """保存训练后的网络"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(bn, f)
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def load_trained_network(filename):
    """加载训练后的网络"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None