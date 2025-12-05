import psutil

# 这里指定只使用这P核
p_cores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #, 16,17,18,19,20,21,22,23

# 遍历所有进程
for proc in psutil.process_iter():
    try:
        # 检查进程名称是否为Python相关
        if proc.name() in ["python.exe", "pythonw.exe", "python3.exe"]:
            # 获取进程ID和名称
            pid = proc.pid
            name = proc.name()
            
            # 设置CPU亲和力
            proc.cpu_affinity(p_cores)
            print(f"已将进程 {pid} ({name}) 绑定到P核 {p_cores}")
    except Exception as e:
        # 忽略没有权限或已结束的进程
        if "permission denied" not in str(e).lower() and "no such process" not in str(e).lower():
            print(f"设置进程 {proc.pid} 时出错: {e}")
