
# import sys, time, traceback
# import utils

# def main():
#     print("🚀 启动程序：Bayesian Network Parameter Learning")
#     if len(sys.argv) < 3:
#         print("用法: python main.py <alarm.bif> <records.dat>")
#         sys.exit(1)

#     bif, rec = sys.argv[1:3]
#     print("🧪 载入网络与数据 ...")
#     t0 = time.time()
#     try:
#         bn, df, mis_idx = utils.setup_network(bif, rec)
#     except Exception as e:
#         print("❌ 初始失败:", e)
#         traceback.print_exc()
#         sys.exit(1)

#     print(f"✅ 完成初始化, 用时 {time.time()-t0:.2f}s")
#     # TODO: 继续实现 EM 推断

# if __name__ == "__main__":
#     main()

import sys, time, traceback
import utils
import EM_debug as EM  # 使用调试版本

def main():
    print("🚀 启动程序：Bayesian Network Parameter Learning (调试模式)")
    if len(sys.argv) < 3:
        print("用法: python main.py <alarm.bif> <records.dat>")
        sys.exit(1)

    bif, rec = sys.argv[1:3]
    
    # 1. 初始化阶段
    print("🧪 载入网络与数据 ...")
    t0 = time.time()
    try:
        bn, df, mis_idx = utils.setup_network(bif, rec)
    except Exception as e:
        print("❌ 初始失败:", e)
        traceback.print_exc()
        sys.exit(1)

    print(f"✅ 完成初始化, 用时 {time.time()-t0:.2f}s")
    
    # 2. 显示数据信息
    complete_records = sum(1 for x in mis_idx if x == -1)
    missing_records = len(df) - complete_records
    print(f"📊 数据规模: {len(df)} 条记录, {len(bn.Pres_Graph)} 个变量")
    print(f"📈 完整记录: {complete_records}, 缺失记录: {missing_records}")
    
    # 3. EM算法阶段
    print("\n" + "="*50)
    print("🔄 开始EM算法参数学习 (调试模式)")
    print("="*50)
    
    em_start = time.time()
    try:
        # 调用调试版本EM算法
        trained_bn, final_likelihood = EM.em_algorithm(
            bn=bn, 
            df=df, 
            mis_idx=mis_idx,
            max_iter=5,  # 减少迭代次数便于调试
            tolerance=1e-4,
            verbose=True
        )
        
        em_time = time.time() - em_start
        print(f"\n🏁 EM算法完成！总用时: {em_time:.2f}s")
        print(f"🎯 最终对数似然: {final_likelihood:.4f}")
        
    except Exception as e:
        print("❌ EM算法失败:", e)
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✨ 调试完成！")

if __name__ == "__main__":
    main()

'''
调用路径

main.py
    ↓ 调用
utils.py
    ↓ 调用
bayesnet.py

main.py → utils.setup_network(bif, rec)

utils.py 内部调用：
    bayesnet.read_network(bif_alarm) - 读取网络结构
    bayesnet.network.set_mb() - 设置马尔可夫毯
    utils.get_data(dat_records) - 处理数据
    utils.init_params() - 初始化参数
    utils.get_missing_index() - 处理缺失数据

bayesnet.py 提供核心功能：
    Graph_Node 类 - 节点数据结构
    network 类 - 网络容器
    read_network() - 文件解析器
'''

'''
返回值说明
bn, df, mis_idx = utils.setup_network(bif, rec)
    bn: 完整的 network 对象，包含所有节点和结构
    df: 处理后的观测数据 DataFrame
    mis_idx: 缺失数据的索引信息
'''

'''
设计模式
这是典型的分层架构：
    main.py: 控制层（程序入口，流程控制）
    utils.py: 服务层（业务逻辑，数据处理）
    bayesnet.py: 数据层（核心数据结构和算法）
'''

'''
初始化过程详解
当运行 bn, df, mis_idx = utils.setup_network(bif, rec) 时，具体执行了以下步骤：

1️⃣ 读取网络结构 (bayesnet.read_network(bif_alarm))
做了什么：

解析 gold_alarm.bif 文件
创建37个 Graph_Node 对象，每个代表一个医疗变量
建立父子关系（如：StrokeVolume 的父节点是 LVFailure 和 Hypovolemia）
加载每个节点的条件概率表(CPT)
将原始概率数据转换为 pandas DataFrame 格式
结果：得到完整的贝叶斯网络结构 bn

2️⃣ 设置马尔可夫毯 (bn.set_mb())
做了什么：

为每个节点计算其马尔可夫毯（父节点+子节点+配偶节点）
存储在 bn.MB 字典中，格式如：
作用：为后续的条件独立性推理和采样提供快速查找

3️⃣ 加载观测数据 (utils.get_data(dat_records))
做了什么：

读取 records.dat 文件（原始文本数据）
设置37个变量的列名
将字符串标签转换为数字编码：
结果：得到数值化的观测数据 df

4️⃣ 初始化参数 (utils.init_params())
做了什么：

为每个节点的CPT设置初始计数值
通常使用均匀先验（每个参数初始化为1.0）
调用 normalise_cpt() 将计数转换为概率分布
作用：为EM算法提供起始参数

5️⃣ 标记缺失数据 (utils.get_missing_index())
做了什么：

扫描每行数据，找出缺失变量
返回 mis_idx 数组，其中：
mis_idx[i] = -1：第i行是完整记录
mis_idx[i] = j：第i行的第j个变量缺失
作用：让EM算法知道哪些数据需要特殊处理

📊 初始化完成后的状态
现在你拥有：

bn - 完整的贝叶斯网络
    37个节点及其关系
    每个节点的初始CPT参数
    马尔可夫毯信息
df - 处理后的数据
    数值编码的观测记录
    缺失值标记为 NaN
    mis_idx - 缺失数据索引

指示每行的缺失模式
🎯 为什么需要这些准备？
这些初始化步骤为EM算法准备了所有必需的组件：

网络结构：知道变量间的依赖关系
初始参数：EM算法的起点
数据格式：统一的数值表示
缺失标记：区分完整和不完整记录
现在可以开始实现EM算法了！
'''
############################################################
#现在开始实现EM算法
############################################################