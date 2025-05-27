
import sys, time, traceback
import utils

def main():
    print("🚀 启动程序：Bayesian Network Parameter Learning")
    if len(sys.argv) < 3:
        print("用法: python main.py <alarm.bif> <records.dat>")
        sys.exit(1)

    bif, rec = sys.argv[1:3]
    print("🧪 载入网络与数据 ...")
    t0 = time.time()
    try:
        bn, df, mis_idx = utils.setup_network(bif, rec)
    except Exception as e:
        print("❌ 初始失败:", e)
        traceback.print_exc()
        sys.exit(1)

    print(f"✅ 完成初始化, 用时 {time.time()-t0:.2f}s")
    # TODO: 继续实现 EM 推断

if __name__ == "__main__":
    main()
