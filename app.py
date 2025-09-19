import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from collections import deque
import os
import time
import csv
import tempfile
from fpdf import FPDF
import threading
import matplotlib.pyplot as plt
import torch
import winsound
try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import beepy
    BEEP_AVAILABLE = True
except ImportError:
    BEEP_AVAILABLE = False

# 導入子模組
try:
    from ui import UIManager
except ImportError as e:
    print(f"導入UI模組失敗: {e}")
    print("請確保ui.py檔案與app.py在同一目錄中")
    exit(1)

try:
    from data_processor import DataProcessor
except ImportError as e:
    print(f"導入資料處理模組失敗: {e}")
    print("請確保data_processor.py檔案與app.py在同一目錄中")
    exit(1)

try:
    from video_processor import VideoProcessor
except ImportError as e:
    print(f"導入影片處理模組失敗: {e}")
    print("請確保video_processor.py檔案與app.py在同一目錄中")
    exit(1)

try:
    from resource_manager import ResourceManager
except ImportError as e:
    print(f"導入資源管理模組失敗: {e}")
    print("請確保resource_manager.py檔案與app.py在同一目錄中")
    exit(1)

class YOLOInferenceApp:
    def __init__(self, root):
        self.root = root
        self.is_monitoring = False
        self.root.title("YOLO 即時推論系統 - Material Area Detection (GPU Accelerated)")
        self.root.geometry("1700x950")
        self.root.configure(bg='#f0f0f0')
        
        # 檢查YOLO是否可用
        if not YOLO_AVAILABLE:
            messagebox.showerror("錯誤", "請安裝 ultralytics: pip install ultralytics")
            return
        
        # 初始化子模組
        self.resource_manager = ResourceManager(self)
        self.video_processor = VideoProcessor(self)
        self.ui_manager = UIManager(self)
        
        # GPU 初始化檢查
        self.resource_manager.setup_gpu()
        
        # 初始化UI管理器
        self.ui_manager.setup_styles()
        
        # 初始化所有變量
        self.model = None
        self.cap = None
        self.video_thread = None
        self.plot_update_thread = None
        self.monitor_thread = None
        self.is_inference = False  # 取代原 is_running，為推論模式
        self.is_preview = False     # 新增預覽模式
        self.current_frame = None
        self.stream_type = "none"  # "video", "camera", "rtsp", "none"
        self.output_video_writer = None
        
        # 面積計算相關變量
        self.raw_areas = []
        self.raw_time_stamps = []
        self.avg_areas = []
        self.avg_time_stamps = []
        self.sma_areas = []  # 平滑後的面積數據
        self.sma_time_stamps = []
        self.frame_group = []
        self.frame_times = []
        self.frame_count = 0
        self.last_detect_time = None
        self.video_fps = 30  # 預設FPS
        
        # 即時數據顯示
        self.recent_areas = deque(maxlen=100)
        self.recent_times = deque(maxlen=100)
        
        # 圖表更新控制
        self.last_plot_update = time.time()
        self.plot_update_interval = 1  # 1秒更新一次
        
        # 監控控制
        self.inference_start_time = None
        self.max_inference_time = 360 * 60  # 12分鐘
        self.max_no_detect_time = 60 * 60   # 5分鐘無檢測
        
        # NG 追蹤
        self.ng_3_times = set()
        self.ng_5_times = set()
        
        # 新增：監控開始時的影像
        self.start_monitor_image = None
        
        # 設定UI
        self.ui_manager.setup_ui()
        self.ui_manager.setup_initial_plot()
        self.load_model(silent=True)
    
    def update_confidence_label(self, value):
        """更新信心閾值標籤"""
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    def update_fps_label(self, value):
        """更新FPS標籤"""
        self.fps_label.config(text=f"{int(float(value))}")
    
    def load_model(self, silent=False):
        """載入YOLO模型並配置GPU"""
        try:
            self.status_var.set("🔄 載入模型中...")
            self.system_info_var.set("正在載入AI模型...")
            model_path = self.model_var.get()
            
            # 檢查自定義模型路徑是否存在
            if not model_path.endswith('.pt') or not any(model_path.endswith(std) for std in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']):
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"找不到模型檔案: {model_path}")
            
            # 載入模型並指定設備
            self.model = YOLO(model_path)
            
            # 配置模型到GPU
            if self.gpu_available:
                self.model.to(device=self.device)
                # 預熱GPU
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                with torch.no_grad():
                    _ = dummy_input * 2  # 簡單的GPU操作來預熱
                device_info = f"GPU-{self.device}"
            else:
                device_info = "CPU"
            
            model_name = model_path.split('\\')[-1] if '\\' in model_path else model_path
            self.status_var.set(f"✅ 模型載入成功 ({device_info})")
            self.system_info_var.set(f"AI模型已就緒: {model_name} ({device_info})")
            
            # 更新GPU記憶體資訊
            if self.gpu_available:
                self.memory_info_var.set(self.resource_manager.get_gpu_memory_info())
            
            if not silent:
                messagebox.showinfo("載入成功", f"YOLO模型載入成功！\n運行裝置: {device_info}")
                
        except Exception as e:
            self.status_var.set("❌ 模型載入失敗")
            self.system_info_var.set("模型載入錯誤")
            messagebox.showerror("載入錯誤", f"載入模型失敗: {str(e)}")
    
    def on_model_change(self, event=None):
        """模型選擇改變時的處理"""
        if self.is_inference:
            messagebox.showwarning("⚠️ 警告", "請先停止當前推論再更換模型")
            return
        self.load_model()
    
    def upload_video(self):
        """上傳影片檔案"""
        if self.is_inference or self.is_preview:
            messagebox.showwarning("⚠️ 警告", "請先停止當前操作")
            return
            
        file_path = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[
                ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("所有檔案", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 釋放舊的 VideoCapture 資源
                if self.cap:
                    self.cap.release()
                    self.cap = None
                    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 已釋放舊的 VideoCapture 資源")
                
                # 嘗試開啟新影片
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise Exception(f"無法開啟影片檔案: {file_path} - 可能格式不支援或檔案損壞")
                
                # 獲取影片屬性
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0:
                    self.video_fps = 30  # 預設值
                    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 警告: 無法獲取影片FPS，預設為 {self.video_fps}")
                
                self.stream_type = "video"
                filename = os.path.basename(file_path)
                self.status_var.set(f"📁 影片已載入: {filename}")
                self.system_info_var.set(f"影片來源: {filename} (FPS: {self.video_fps:.1f})")
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 成功載入影片: {filename}")
                # 啟動預覽
                self.start_preview()
            except Exception as e:
                self.cap = None  # 確保 cap 被重置
                messagebox.showerror("載入錯誤", f"載入影片失敗: {str(e)}")
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 錯誤細節 - 載入影片 {file_path}: {str(e)}")

    def open_camera(self):
        """開啟本地攝像機"""
        if self.is_inference or self.is_preview:
            messagebox.showwarning("⚠️ 警告", "請先停止當前操作")
            return
            
        try:
            camera_id = int(self.camera_id_var.get())
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise Exception(f"無法開啟攝像機 ID: {camera_id}")
            self.stream_type = "camera"
            self.status_var.set(f"📷 本地攝像機已開啟")
            self.system_info_var.set(f"攝像機來源: 本地攝像機 #{camera_id}")
            # 啟動預覽
            self.start_preview()
        except Exception as e:
            messagebox.showerror("連接錯誤", f"開啟攝像機失敗: {str(e)}")
    
    def open_rtsp_stream(self):
        """開啟RTSP串流"""
        if self.is_inference or self.is_preview:
            messagebox.showwarning("⚠️ 警告", "請先停止當前操作")
            return
            
        try:
            rtsp_url = self.rtsp_url_var.get().strip()
            if not rtsp_url:
                raise Exception("請輸入RTSP URL")
                
            if self.cap:
                self.cap.release()
                
            # 設定OpenCV參數以提高RTSP串流穩定性
            self.cap = cv2.VideoCapture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝延遲
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # 設定幀率
            
            if not self.cap.isOpened():
                raise Exception("無法連接到RTSP串流，請檢查URL和網路連線")
                
            self.stream_type = "rtsp"
            self.status_var.set("🌐 RTSP串流已連接")
            self.system_info_var.set("串流來源: RTSP網路攝像機")
            # 啟動預覽
            self.start_preview()
        except Exception as e:
            messagebox.showerror("連接錯誤", f"連接RTSP串流失敗: {str(e)}")
    
    def start_preview(self):
        """啟動影像預覽"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("警告", "無有效的影片來源，無法啟動預覽")
            return
        self.is_preview = True
        self.video_thread = threading.Thread(target=self.video_processor.process_video, daemon=True)
        self.video_thread.start()
        self.status_var.set("📺 預覽模式執行中...")
        self.system_info_var.set("影像預覽中 (無推論)")
        self.memory_info_var.set(self.resource_manager.get_gpu_memory_info())
    
    def toggle_monitoring(self):
        """切換監控狀態並擷取開始監控時的影像"""
        if not self.is_inference:
            messagebox.showwarning("⚠️ 警告", "請先開始推論再啟動監控")
            return
        
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.monitor_button.config(text="🛑 停止監控")
            self.status_var.set("🚨 監控已啟動")
            # 擷取開始監控時的影像
            if self.current_frame is not None:
                self.start_monitor_image = self.current_frame.copy()
            else:
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Warning: No current frame available for start monitor image")
        else:
            self.monitor_button.config(text="🔍 開始監控")
            self.status_var.set("🟢 監控已停止")
            self.start_monitor_image = None
    
    def start_inference(self):
        """開始推論"""
        if not self.model:
            messagebox.showerror("模型錯誤", "請先載入YOLO模型")
            return
            
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("來源錯誤", "請先選擇影片來源")
            return
        
        # 重置計數器和數據
        self.reset_area_data()
        
        # 清理GPU記憶體
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        # 設定輸出影片
        if self.save_output_var.get() and self.stream_type == "video":
            self.setup_output_video()
        
        self.is_inference = True
        self.inference_start_time = time.time()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.DISABLED)
        
        # 如果預覽中，無需重新啟動線程，直接切換模式
        if self.video_thread is None or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=self.video_processor.process_video, daemon=True)
            self.video_thread.start()
        
        # 啟動圖表更新線程
        self.plot_update_thread = threading.Thread(target=self.plot_update_loop, daemon=True)
        self.plot_update_thread.start()
        
        # 啟動監控線程
        self.monitor_thread = threading.Thread(target=self.monitor_inference, daemon=True)
        self.monitor_thread.start()
        
        device_info = "GPU" if self.gpu_available else "CPU"
        self.status_var.set(f"🚀 推論執行中... ({device_info})")
        self.system_info_var.set("AI模型正在分析中...")
        self.memory_info_var.set(self.resource_manager.get_gpu_memory_info())
    
    def stop_inference(self):
        """停止推論，切換回預覽模式"""
        self.is_inference = False
        self.is_monitoring = False
        self.monitor_button.config(text="🔍 開始監控")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.NORMAL)
        
        # 清理GPU記憶體
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        # 關閉輸出影片
        if self.output_video_writer:
            self.output_video_writer.release()
            self.output_video_writer = None
        
        self.status_var.set("📺 預覽模式")
        self.system_info_var.set("系統待機中 (預覽繼續)")
        self.fps_var.set("🎯 FPS: 0")
        self.runtime_var.set("⏱️ Runtime: 00:00")
        self.memory_info_var.set(self.resource_manager.get_gpu_memory_info())
        # 保持預覽模式
        self.is_preview = True
    
    def stop_processing(self):
        """完全停止所有處理並釋放資源"""
        self.is_inference = False
        self.is_preview = False
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.output_video_writer:
            self.output_video_writer.release()
            self.output_video_writer = None
        
        # 清理GPU記憶體
        if self.gpu_available:
            torch.cuda.empty_cache()
            
        self.status_var.set("⏹️ 已停止")
        self.system_info_var.set("系統待機中")
        self.fps_var.set("🎯 FPS: 0")
        self.runtime_var.set("⏱️ Runtime: 00:00")
        self.memory_info_var.set(self.resource_manager.get_gpu_memory_info())
    
    def clear_all_data(self):
        """清除所有資料並重置系統"""
        if self.is_inference or self.is_preview:
            result = messagebox.askyesno("確認清除", "系統正在運行中，是否要停止並清除所有資料？")
            if not result:
                return
            self.stop_processing()
        
        try:
            # 清除所有數據
            self.reset_area_data()
            
            # 清理資源
            self.resource_manager.clear_memory_and_resources()
            
            # 重置UI顯示
            self.current_area_var.set("📐 當前面積: --")
            self.avg_area_var.set("📊 平均面積: --") 
            self.sma_area_var.set("📈 SMA面積: --")
            self.fps_var.set("🎯 FPS: 0")
            self.runtime_var.set("⏱️ Runtime: 00:00")
            
            # 重置圖表
            self.reset_sma_plot()
            
            # 重置影像顯示
            self.image_label.configure(text="🎬 請選擇影片來源開始檢測", image='')
            self.image_label.image = None
            
            self.status_var.set("🟡 資料已清除，系統重置完成")
            self.system_info_var.set("✅ 清除完成")
            self.memory_info_var.set(self.resource_manager.get_gpu_memory_info())
            
            messagebox.showinfo("清除完成", "所有資料已清除，系統已重置！")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"清除資料時發生錯誤: {str(e)}")
    
    def reset_area_data(self):
        """重置面積計算數據"""
        self.raw_areas = []
        self.raw_time_stamps = []
        self.avg_areas = []
        self.avg_time_stamps = []
        self.sma_areas = []
        self.sma_time_stamps = []
        self.frame_group = []
        self.frame_times = []
        self.frame_count = 0
        self.last_detect_time = None
        self.recent_areas.clear()
        self.recent_times.clear()
        self.last_plot_update = time.time()
        self.ng_3_times = set()
        self.ng_5_times = set()
        self.start_monitor_image = None
    
    def setup_output_video(self):
        """設定輸出影片"""
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_fps
            
            output_path = filedialog.asksaveasfilename(
                title="儲存輸出影片",
                defaultextension=".mp4",
                filetypes=[("MP4 檔案", "*.mp4"), ("所有檔案", "*.*")]
            )
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.output_video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except Exception as e:
            messagebox.showwarning("設定警告", f"設定輸出影片失敗: {str(e)}")
            self.save_output_var.set(False)
    
    def monitor_inference(self):
        """監控推論狀態，自動重置"""
        while self.is_inference:
            time.sleep(10)  # 每10秒檢查一次
            
            if not self.is_inference:
                break
                
            current_time = time.time()
            
            # 更新GPU記憶體資訊
            if self.gpu_available:
                self.root.after(0, lambda: self.memory_info_var.set(self.resource_manager.get_gpu_memory_info()))
            
            # 檢查總運行時間是否超過12分鐘
            if self.inference_start_time and (current_time - self.inference_start_time) > self.max_inference_time:
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 推論時間超過120分鐘，自動重置...")
                self.root.after(0, self.auto_reset_system)
                break
            
            # 檢查是否超過5分鐘沒有檢測到物件
            if self.last_detect_time and (current_time - self.last_detect_time) > self.max_no_detect_time:
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 超過5分鐘無檢測，自動重置...")
                self.root.after(0, self.auto_reset_system)
                break
    
    def auto_reset_system(self):
        """自動重置系統並釋放資源"""
        self.stop_inference()
        self.resource_manager.clear_memory_and_resources()
        self.reset_sma_plot()
        self.status_var.set("🔄 系統自動重置: 檢測超時")
        self.system_info_var.set("⚠️ 自動重置完成")
        messagebox.showinfo("自動重置", "系統已自動重置\n原因: 12分鐘運行超時或5分鐘無檢測")
    
    def reset_sma_plot(self):
        """重置SMA圖表"""
        try:
            self.ax.clear()
            self.ui_manager.setup_initial_plot()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 重置圖表時發生錯誤: {e}")
    
    def plot_update_loop(self):
        """每1秒更新一次圖表的循環"""
        while self.is_inference:
            time.sleep(1)  # 每秒檢查一次
            current_time = time.time()
            
            if current_time - self.last_plot_update >= self.plot_update_interval:
                if len(self.sma_areas) > 0:
                    self.root.after(0, self.update_sma_plot)
                self.last_plot_update = current_time
    
    def update_sma_plot(self):
        """更新SMA面積趨勢圖"""
        try:
            self.ax.clear()
            
            # 設定圖表樣式
            self.ax.set_facecolor('#fafafa')
            
            if len(self.sma_areas) > 0 and len(self.sma_time_stamps) > 0:
                # 過濾有效數據
                valid_data = [(t, a) for t, a in zip(self.sma_time_stamps, self.sma_areas) if not np.isnan(a)]
                
                if valid_data:
                    times, areas = zip(*valid_data)
                    
                    # 繪製SMA曲線
                    self.ax.plot(times, areas, linewidth=3, color='#e74c3c', alpha=0.9, 
                               label=f'SMA Area (Window: {self.sma_window_var.get()})')
                    
                    # 填充區域
                    self.ax.fill_between(times, areas, alpha=0.25, color='#e74c3c')
                    
                    # 數據點標記
                    self.ax.scatter(times, areas, color='#c0392b', s=25, alpha=0.8, zorder=5)
                    
                    # 標註最新值
                    if len(times) > 0:
                        latest_time, latest_area = times[-1], areas[-1]
                        self.ax.annotate(f'{latest_area:.0f} px²', 
                                       xy=(latest_time, latest_area), 
                                       xytext=(15, 15), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1c40f', 
                                               edgecolor='#f39c12', alpha=0.9),
                                       fontsize=10, fontweight='bold', color='#2c3e50',
                                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                                     color='#f39c12', alpha=0.7))
                    
                    # 統計資訊
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    max_area = np.max(areas)
                    min_area = np.min(areas)
                    
                    stats_text = f'📈 Statistics\n'
                    stats_text += f'Mean: {mean_area:.0f} px²\n'
                    stats_text += f'Std: {std_area:.0f} px²\n'
                    stats_text += f'Max: {max_area:.0f} px²\n'
                    stats_text += f'Min: {min_area:.0f} px²'
                    
                    # 添加GPU狀態
                    if self.gpu_available:
                        stats_text += f'\n\n🔥 GPU Accelerated'
                    
                    self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                       edgecolor='#bdc3c7', alpha=0.9),
                               fontsize=8, color='#2c3e50')
                    
                    # 趨勢指標
                    if len(areas) >= 2:
                        trend = "📈" if areas[-1] > areas[-2] else "📉" if areas[-1] < areas[-2] else "➡️"
                        trend_text = f"{trend} Trend"
                        self.ax.text(0.98, 0.98, trend_text, transform=self.ax.transAxes,
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.8),
                                   fontsize=9, fontweight='bold')
                    
                    # 標記NG
                    # 標記預警 (3分鐘 NG)
                    for t in self.ng_3_times:
                        self.ax.axvline(t, color='orange', linestyle='--', alpha=0.7)
                        self.ax.text(t, 600000 * 0.95, '預警', rotation=90, va='top', ha='right', color='orange', fontsize=8)
                    
                    # 標記警報 (5分鐘 NG)
                    for t in self.ng_5_times:
                        self.ax.axvline(t, color='red', linestyle='-', alpha=0.8)
                        self.ax.text(t, 600000 * 0.9, '警報', rotation=90, va='top', ha='right', color='red', fontsize=8)
                        
                else:
                    # 無有效數據時的顯示
                    self.ax.text(0.5, 0.5, '⚠️ No Valid Detection Data\n等待檢測數據...', 
                                transform=self.ax.transAxes, ha='center', va='center', 
                                fontsize=12, color='#e67e22',
                                bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffeaa7', alpha=0.8))
            else:
                # 無數據時的顯示
                init_text = '⏳ Initializing...\n系統準備中'
                if self.gpu_available:
                    init_text += '\n🔥 GPU Ready'
                
                self.ax.text(0.5, 0.5, init_text, 
                            transform=self.ax.transAxes, ha='center', va='center', 
                            fontsize=12, color='#74b9ff',
                            bbox=dict(boxstyle='round,pad=0.8', facecolor='#dde8fc', alpha=0.8))
            
            # 設定圖表標題和標籤
            title_text = '📊 SMA Area Trend Analysis\n(Real-time updates every 1s)'
            if self.gpu_available:
                title_text += ' - GPU Accelerated'
            
            self.ax.set_title(title_text, 
                             fontsize=12, fontweight='bold', color='#2c3e50', pad=20)
            self.ax.set_xlabel('Time (seconds)', fontsize=10, color='#34495e', fontweight='bold')
            self.ax.set_ylabel('Area (px²)', fontsize=10, color='#34495e', fontweight='bold')
            
            # 設定y軸範圍和刻度
            self.ax.set_ylim(0, 600000)
            self.ax.set_yticks(np.arange(0, 600001, 50000))
            
            # 網格和樣式
            self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#bdc3c7')
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_color('#7f8c8d')
            self.ax.spines['bottom'].set_color('#7f8c8d')
            
            # 圖例設定
            if len(self.sma_areas) > 0:
                valid_data = [(t, a) for t, a in zip(self.sma_time_stamps, self.sma_areas) if not np.isnan(a)]
                if valid_data:
                    self.ax.legend(loc='upper left', fontsize=9, framealpha=0.9, 
                                  fancybox=True, shadow=True)
            
            plt.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 更新圖表時發生錯誤: {e}")
    
    def check_and_handle_ng(self):
        """檢查NG並處理警報"""
        cleaned_areas = DataProcessor.clean_data(self.sma_areas, self.sma_time_stamps, gap_limit=3)
        results_3 = DataProcessor.check_area_trend_timebased(cleaned_areas, self.sma_time_stamps, minutes=3, epsilon=0.15, cooldown=180)
        results_5 = DataProcessor.check_area_trend_timebased(cleaned_areas, self.sma_time_stamps, minutes=4, epsilon=0.05, cooldown=210)
        ng_3 = DataProcessor.extract_ng_markers(results_3)
        ng_5 = DataProcessor.extract_ng_markers(results_5)
        
        # 更新NG集合
        self.ng_3_times.update(ng_3)
        new_ng_5 = set(ng_5) - self.ng_5_times
        self.ng_5_times.update(ng_5)
        
        # 處理新5分鐘警報
        for t in new_ng_5:
            # 寫入 log CSV (使用固定檔案名稱 'ng_log.csv')
            start_str = time.strftime("%Y%m%d%H%M", time.localtime(self.inference_start_time))
            ng_str = time.strftime("%Y%m%d%H%M", time.localtime(self.inference_start_time + t))
            range_str = f"{start_str}-{ng_str}"
            duration = 5 * 60
            log_file = 'ng_log.csv'
            file_exists = os.path.exists(log_file) and os.path.getsize(log_file) > 0
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Time Range', 'Duration (seconds)'])
                writer.writerow([range_str, duration])
            
            # 顯示非模態警告視窗並生成PDF
            self.root.after(0, lambda rs=range_str: self.show_non_modal_warning(rs))
    
    def play_beep(self):
        """在獨立執行緒中播放蜂鳴聲"""
        winsound.Beep(2500, 5000)

    def show_non_modal_warning(self, range_str):
        """顯示非模態警告視窗，觸發蜂鳴聲和PDF生成"""
        # 播放蜂鳴聲
        threading.Thread(target=self.play_beep, daemon=True).start()
        
        # 獲取NG時的影像
        ng_image = None
        if self.current_frame is not None:
            ng_image = self.current_frame.copy()
        else:
            print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Warning: No current frame available for NG image")
        
        # 生成PDF（如果有開始監控影像和NG影像）
        if self.start_monitor_image is not None and ng_image is not None:
            self.generate_pdf(range_str, self.start_monitor_image, ng_image)
        else:
            print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Warning: Missing images for PDF generation")
        
        # 顯示警告視窗
        warning_win = tk.Toplevel(self.root)
        warning_win.title("警報")
        warning_win.geometry("300x150")
        warning_win.attributes('-topmost', True)  # 置頂顯示
        warning_win.configure(bg='#f0f0f0')  # 灰色背景
        
        label = tk.Label(warning_win, text="已發生架橋現象,請立即做處理", font=('Microsoft JhengHei UI', 12), bg='#f0f0f0', fg='#2c3e50')
        label.pack(expand=True, pady=20)
        
        close_button = tk.Button(warning_win, text="關閉", command=warning_win.destroy)
        close_button.pack(pady=10)
    
    def generate_pdf(self, range_str, img1, img2):
        """使用FPDF生成包含兩張影像的PDF，優化臨時檔案管理"""
        pdf_filename = f"{range_str}.pdf"
        
        # 使用臨時檔案來儲存影像
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img1, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img2:
            
            # 儲存影像到臨時檔案
            cv2.imwrite(temp_img1.name, img1)
            cv2.imwrite(temp_img2.name, img2)
            
            # 建立PDF
            pdf = FPDF()
            pdf.add_page()
            
            # 添加影像到PDF
            pdf.image(temp_img1.name, x=10, y=10, w=190, h=100)
            pdf.image(temp_img2.name, x=10, y=120, w=190, h=100)
            
            # 儲存PDF
            pdf.output(pdf_filename)
        
        # 清理臨時檔案
        os.remove(temp_img1.name)
        os.remove(temp_img2.name)
        
        print(f"[{time.strftime('%H:%M:%S', time.localtime())}] PDF 已生成: {pdf_filename}")

def main():
    root = tk.Tk()
    app = YOLOInferenceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.resource_manager.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()