"""
修正版实时监控脚本 - GPU (NVML) 和 CPU
"""
import os
import time
import psutil
import torch
import pynvml  # 引入 NVML 库
from datetime import datetime

def monitor_system():
    print("=" * 80)
    print("系统资源监控 (NVML版) - 按 Ctrl+C 停止")
    print("=" * 80)

    # 1. 初始化 NVIDIA 驱动接口
    nvml_available = False
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 默认监控 GPU 0
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        # 兼容性处理：某些版本返回 bytes，需要解码
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
            
        print(f"GPU: {gpu_name}")
        nvml_available = True
    except Exception as e:
        print(f"GPU: 无法连接 NVIDIA 驱动 ({e})")
        print("将仅监控 CPU/RAM")

    print("-" * 80)

    try:
        while True:
            # 当前时间
            now = datetime.now().strftime("%H:%M:%S")

            # 2. CPU 监控 (interval=None 实现非阻塞，瞬间返回上次调用后的平均值)
            # 第一次调用可能返回 0.0，但在循环中是准的
            cpu_percent = psutil.cpu_percent(interval=None) 
            cpu_count = psutil.cpu_count()

            # 3. 内存监控
            memory = psutil.virtual_memory()
            mem_used = memory.used / 1024**3
            mem_total = memory.total / 1024**3
            mem_percent = memory.percent

            # 4. GPU 监控 (读取系统全局状态)
            gpu_info = ""
            if nvml_available:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # info.used 包含所有进程占用的显存 (类似 nvidia-smi)
                    gpu_used = info.used / 1024**2  # MB
                    gpu_total = info.total / 1024**2 # MB
                    gpu_percent = (gpu_used / gpu_total) * 100
                    
                    # 获取 GPU 利用率 (Load)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu

                    gpu_info = (
                        f" | GPU Mem: {gpu_used:.0f}/{gpu_total:.0f} MB ({gpu_percent:.1f}%) "
                        f"Load: {gpu_util}%"
                    )
                except:
                    gpu_info = " | GPU: 读取失败"

            # 输出
            print(f"\r[{now}] "
                  f"CPU: {cpu_percent:5.1f}% | "
                  f"RAM: {mem_used:.1f}/{mem_total:.1f} GB ({mem_percent:.1f}%)"
                  f"{gpu_info}",
                  end='', flush=True)

            time.sleep(1) # 真正的 1 秒刷新

    except KeyboardInterrupt:
        print("\n\n监控已停止")
        if nvml_available:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    monitor_system()