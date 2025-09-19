import torch
import gc
import cv2
import matplotlib.pyplot as plt
from tkinter import messagebox
import time
class ResourceManager:
    def __init__(self, app):
        self.app = app

    def setup_gpu(self):
        """設定並檢查GPU配置"""
        try:
            # 檢查CUDA是否可用
            if torch.cuda.is_available():
                self.app.gpu_available = True
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    self.app.device = 0  # 使用第一個GPU的索引
                    gpu_name = torch.cuda.get_device_name(self.app.device)
                    gpu_memory = torch.cuda.get_device_properties(self.app.device).total_memory / 1024**3  # GB
                    self.app.gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
                    print(f"[{time.strftime('%H:%M:%S')}] GPU 初始化成功: {self.app.gpu_info}")
                else:
                    self.app.device = 'cpu'
                    self.app.gpu_available = False
                    self.app.gpu_info = "GPU: 無可用CUDA設備，使用CPU"
                    messagebox.showwarning("GPU 警告", "未偵測到可用CUDA設備，將使用CPU")
            else:
                self.app.device = 'cpu'
                self.app.gpu_available = False
                self.app.gpu_info = "GPU: 未偵測到CUDA支援，使用CPU"
                messagebox.showwarning("GPU 警告", "未偵測到CUDA支援的GPU，將使用CPU")

            # 設定GPU記憶體增長策略
            if self.app.gpu_available:
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True

        except Exception as e:
            self.app.device = 'cpu'
            self.app.gpu_available = False
            self.app.gpu_info = f"GPU: 初始化失敗 - {str(e)}"
            messagebox.showerror("GPU 錯誤", f"GPU初始化失敗: {str(e)}\n將使用CPU運算")

    def get_gpu_memory_info(self):
        """獲取GPU記憶體使用資訊"""
        if self.app.gpu_available:
            try:
                allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
                return f"GPU記憶體: {allocated:.1f}GB / {reserved:.1f}GB"
            except:
                return "GPU記憶體: 無法讀取"
        else:
            return "記憶體: CPU模式"

    def clear_memory_and_resources(self):
        """清理記憶體和GPU資源"""
        try:
            # 清理YOLO模型記憶體
            if self.app.model:
                del self.app.model
                self.app.model = None
            
            # 清理所有數據列表
            self.app.raw_areas.clear()
            self.app.raw_time_stamps.clear()
            self.app.avg_areas.clear() 
            self.app.avg_time_stamps.clear()
            self.app.sma_areas.clear()
            self.app.sma_time_stamps.clear()
            self.app.frame_group.clear()
            self.app.frame_times.clear()
            self.app.recent_areas.clear()
            self.app.recent_times.clear()
            
            # 強制垃圾回收
            gc.collect()
            
            # 清理GPU緩存
            if self.app.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            print(f"[{time.strftime('%H:%M:%S')}] 記憶體和GPU資源已清理")
            self.app.memory_info_var.set(self.get_gpu_memory_info())
            
            # 重新載入模型
            self.app.load_model(silent=True)
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 清理資源時發生錯誤: {e}")

    def on_closing(self):
        """關閉應用程式時的清理工作"""
        self.app.stop_processing()
        
        # 清理資源
        try:
            if hasattr(self.app, 'model') and self.app.model:
                del self.app.model
            gc.collect()
            if self.app.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        cv2.destroyAllWindows()
        plt.close('all')
        self.app.root.destroy()