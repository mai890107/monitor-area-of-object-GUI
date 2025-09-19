import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import time

class UIManager:
    """UI介面管理類別 - 負責所有介面元素的建立和配置"""
    
    def __init__(self, app_instance):
        """初始化UI管理器
        
        Args:
            app_instance: YOLOInferenceApp的實例
        """
        self.app = app_instance
        self.root = app_instance.root
        
    def setup_styles(self):
        """設定UI樣式主題"""
        style = ttk.Style()
        
        # 設定現代化主題
        style.theme_use('clam')
        
        # 自定義樣式
        style.configure('Title.TLabelframe', font=('Microsoft JhengHei UI', 11, 'bold'))
        style.configure('Control.TLabelframe', font=('Microsoft JhengHei UI', 10, 'bold'), foreground='#2c3e50')
        style.configure('Modern.TButton', font=('Microsoft JhengHei UI', 9), padding=6)
        style.configure('Action.TButton', font=('Microsoft JhengHei UI', 9, 'bold'))
        style.configure('Stop.TButton', font=('Microsoft JhengHei UI', 9, 'bold'))
        style.configure('Clear.TButton', font=('Microsoft JhengHei UI', 9, 'bold'))
        
        # 配色方案
        style.map('Modern.TButton',
                 background=[('active', '#3498db'), ('!active', '#ecf0f1')],
                 foreground=[('active', 'white'), ('!active', '#2c3e50')])
        
        style.map('Action.TButton',
                 background=[('active', '#27ae60'), ('!active', '#2ecc71')],
                 foreground=[('active', 'white'), ('!active', 'white')])
        
        style.map('Stop.TButton',
                 background=[('active', '#c0392b'), ('!active', '#e74c3c')],
                 foreground=[('active', 'white'), ('!active', 'white')])
        
        style.map('Clear.TButton',
                 background=[('active', '#f39c12'), ('!active', '#f1c40f')],
                 foreground=[('active', 'white'), ('!active', '#2c3e50')])
    
    def setup_ui(self):
        """設定主要UI介面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置網格權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, minsize=320, weight=0)  # 控制面板
        main_frame.columnconfigure(1, weight=3, minsize=600)  # 影像區域 
        main_frame.columnconfigure(2, weight=2, minsize=450)  # 圖表區域
        main_frame.rowconfigure(0, weight=1)
        
        # 建立各區塊
        self._setup_control_panel(main_frame)
        self._setup_display_area(main_frame)
        self._setup_chart_area(main_frame)
        self.setup_status_dashboard(main_frame)
        
    def _setup_control_panel(self, parent):
        """設定左側控制面板"""
        # 創建一個簡單的Frame，使用pack布局
        control_frame = ttk.LabelFrame(parent, text="🎛️ 控制面板", style='Control.TLabelframe', padding="12")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8))
        
        # 創建滾動區域
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        def _configure_scrollable_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
            
        scrollable_frame.bind("<Configure>", _configure_scrollable_frame)
        canvas.bind("<Configure>", _configure_canvas)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # === 建立各個區塊 ===
        self._setup_model_section(scrollable_frame)
        self._setup_source_section(scrollable_frame)
        self._setup_rtsp_section(scrollable_frame)
        self._setup_inference_section(scrollable_frame)
        self._setup_button_section(scrollable_frame)
        self._setup_info_section(scrollable_frame)
        
    def _setup_model_section(self, parent):
        """設定模型選擇區塊"""
        model_section = ttk.LabelFrame(parent, text="🤖 模型設定", padding="8")
        model_section.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(model_section, text="YOLO 模型:", font=('Microsoft JhengHei UI', 9)).pack(anchor=tk.W, pady=2)
        self.app.model_var = tk.StringVar(value="models/area0903.pt")
        model_combo = ttk.Combobox(model_section, textvariable=self.app.model_var, 
                                  values=["models/area0903.pt",
                                         "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                                  font=('Consolas', 8))
        model_combo.pack(fill=tk.X, pady=2)
        model_combo.bind("<<ComboboxSelected>>", self.app.on_model_change)
        
        ttk.Button(model_section, text="🔄 載入模型", command=self.app.load_model, style='Modern.TButton').pack(fill=tk.X, pady=3)
        
    def _setup_source_section(self, parent):
        """設定影片來源區塊"""
        source_section = ttk.LabelFrame(parent, text="📹 影片來源", padding="8")
        source_section.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Button(source_section, text="📁 上傳影片", command=self.app.upload_video, style='Modern.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(source_section, text="📷 本地攝像機", command=self.app.open_camera, style='Modern.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(source_section, text="🌐 RTSP串流", command=self.app.open_rtsp_stream, style='Modern.TButton').pack(fill=tk.X, pady=2)
        
        # 攝像機ID設定
        camera_frame = ttk.Frame(source_section)
        camera_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(camera_frame, text="攝像機 ID:", font=('Microsoft JhengHei UI', 8)).pack(side=tk.LEFT)
        self.app.camera_id_var = tk.StringVar(value="0")
        camera_spin = tk.Spinbox(camera_frame, from_=0, to=9, width=8, textvariable=self.app.camera_id_var, 
                                font=('Microsoft JhengHei UI', 9))
        camera_spin.pack(side=tk.RIGHT)
        
    def _setup_rtsp_section(self, parent):
        """設定RTSP區塊"""
        rtsp_section = ttk.LabelFrame(parent, text="🔗 RTSP 設定", padding="6")
        rtsp_section.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(rtsp_section, text="RTSP URL:", font=('Microsoft JhengHei UI', 8)).pack(anchor=tk.W)
        self.app.rtsp_url_var = tk.StringVar(value="rtsp://admin:dh123456@192.168.50.154:554/cam/realmonitor?channel=4&subtype=1")
        rtsp_entry = ttk.Entry(rtsp_section, textvariable=self.app.rtsp_url_var, 
                              font=('Consolas', 8))
        rtsp_entry.pack(fill=tk.X, pady=3)
        
    def _setup_inference_section(self, parent):
        """設定推論參數區塊"""
        inference_section = ttk.LabelFrame(parent, text="⚙️ 推論設定", padding="8")
        inference_section.pack(fill=tk.X, pady=(0, 8))
        
        # 信心閾值設定
        conf_frame = ttk.Frame(inference_section)
        conf_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(conf_frame, text="信心閾值:", font=('Microsoft JhengHei UI', 9)).pack(side=tk.LEFT)
        self.app.confidence_var = tk.DoubleVar(value=0.68)
        
        self.app.confidence_label = ttk.Label(conf_frame, text="0.68", font=('Microsoft JhengHei UI', 9, 'bold'), foreground='#e74c3c')
        self.app.confidence_label.pack(side=tk.RIGHT)
        
        confidence_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.app.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X, pady=2)
        confidence_scale.configure(command=self.app.update_confidence_label)
        
        # NaN間隔設定
        nan_frame = ttk.Frame(inference_section)
        nan_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(nan_frame, text="NaN 間隔 (秒):", font=('Microsoft JhengHei UI', 8)).pack(side=tk.LEFT)
        self.app.nan_gap_var = tk.DoubleVar(value=10.0)
        nan_gap_spin = tk.Spinbox(nan_frame, from_=1, to=30, increment=1, textvariable=self.app.nan_gap_var, 
                                 width=8, font=('Microsoft JhengHei UI', 9))
        nan_gap_spin.pack(side=tk.RIGHT)
        
        # SMA窗口設定
        sma_frame = ttk.Frame(inference_section)
        sma_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(sma_frame, text="SMA Window Size:", font=('Microsoft JhengHei UI', 8)).pack(side=tk.LEFT)
        self.app.sma_window_var = tk.IntVar(value=7)
        sma_spin = tk.Spinbox(sma_frame, from_=3, to=21, increment=2, textvariable=self.app.sma_window_var, 
                             width=8, font=('Microsoft JhengHei UI', 9))
        sma_spin.pack(side=tk.RIGHT)
        
        # 輸出選項
        self.app.save_output_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(inference_section, text="💾 儲存輸出影片", variable=self.app.save_output_var).pack(anchor=tk.W, pady=3)
        
        # Inference FPS 設定 (改為滑桿)
        fps_frame = ttk.Frame(inference_section)
        fps_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(fps_frame, text="Inference FPS:", font=('Microsoft JhengHei UI', 8)).pack(side=tk.LEFT)
        self.app.inference_fps_var = tk.IntVar(value=30)
        
        self.app.fps_label = ttk.Label(fps_frame, text="30", font=('Microsoft JhengHei UI', 9, 'bold'), foreground='#3498db')
        self.app.fps_label.pack(side=tk.RIGHT)
        
        fps_scale = ttk.Scale(fps_frame, from_=1, to=60, variable=self.app.inference_fps_var, orient=tk.HORIZONTAL)
        fps_scale.pack(fill=tk.X, pady=2)
        fps_scale.configure(command=self.app.update_fps_label)
        
    def _setup_button_section(self, parent):
        """設定控制按鈕區塊"""
        button_section = ttk.LabelFrame(parent, text="🎮 操作控制", padding="8")
        button_section.pack(fill=tk.X, pady=(0, 8))
        
        # 主要控制按鈕 - 調整間距和padding
        self.app.start_button = ttk.Button(button_section, text="▶️ 開始推論", command=self.app.start_inference, style='Action.TButton')
        self.app.start_button.pack(fill=tk.X, pady=2)
        
        self.app.stop_button = ttk.Button(button_section, text="⏹️ 停止推論", command=self.app.stop_inference, 
                                     state=tk.DISABLED, style='Stop.TButton')
        self.app.stop_button.pack(fill=tk.X, pady=2)
        
        self.app.monitor_button = ttk.Button(button_section, text="🔍 開始監控", command=self.app.toggle_monitoring, style='Action.TButton')
        self.app.monitor_button.pack(fill=tk.X, pady=2)
        
        self.app.clear_button = ttk.Button(button_section, text="🧹 清除資料", command=self.app.clear_all_data, style='Clear.TButton')
        self.app.clear_button.pack(fill=tk.X, pady=2)
        
    def _setup_info_section(self, parent):
        """設定系統資訊區塊"""
        info_section = ttk.LabelFrame(parent, text="📊 系統資訊", padding="8")
        info_section.pack(fill=tk.X, pady=(0, 10))
        
        # GPU資訊顯示
        self.app.gpu_info_var = tk.StringVar(value=self.app.gpu_info)
        gpu_label = ttk.Label(info_section, textvariable=self.app.gpu_info_var, 
                             font=('Microsoft JhengHei UI', 8), 
                             foreground='#27ae60' if self.app.gpu_available else '#e74c3c')
        gpu_label.pack(anchor=tk.W, pady=2)
        
        # 系統狀態顯示
        self.app.system_info_var = tk.StringVar(value="系統就緒")
        info_label = ttk.Label(info_section, textvariable=self.app.system_info_var, 
                              font=('Microsoft JhengHei UI', 8), foreground='#7f8c8d')
        info_label.pack(anchor=tk.W, pady=2)
        
        self.app.memory_info_var = tk.StringVar(value=self.app.resource_manager.get_gpu_memory_info())
        memory_label = ttk.Label(info_section, textvariable=self.app.memory_info_var, 
                                font=('Microsoft JhengHei UI', 8), foreground='#27ae60')
        memory_label.pack(anchor=tk.W, pady=2)
        
    def _setup_display_area(self, parent):
        """設定中間影像顯示區域"""
        display_frame = ttk.LabelFrame(parent, text="📺 即時影像顯示", style='Title.TLabelframe', padding="8")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=8)
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 影像容器
        image_container = ttk.Frame(display_frame, relief='solid', borderwidth=2)
        image_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        image_container.columnconfigure(0, weight=1)
        image_container.rowconfigure(0, weight=1)
        
        # 影像顯示標籤
        self.app.image_label = ttk.Label(image_container, text="🎬 請選擇影片來源開始檢測", 
                                    font=('Microsoft JhengHei UI', 14), anchor='center',
                                    background='#ecf0f1', foreground='#7f8c8d')
        self.app.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
    def _setup_chart_area(self, parent):
        """設定右側圖表區域"""
        chart_frame = ttk.LabelFrame(parent, text="📈 SMA 趨勢分析", style='Title.TLabelframe', padding="8")
        chart_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(8, 0))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # 圖表容器
        chart_container = ttk.Frame(chart_frame, relief='solid', borderwidth=1)
        chart_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3, pady=3)
        chart_container.columnconfigure(0, weight=1)
        chart_container.rowconfigure(0, weight=1)
        
        # 建立matplotlib圖表
        self._setup_matplotlib_chart(chart_container)
        
    def _setup_matplotlib_chart(self, container):
        """設定matplotlib圖表"""
        try:
            plt.style.use('default')
        except:
            pass  # 如果樣式不可用，使用預設
            
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['axes.labelsize'] = 9
        
        self.app.fig, self.app.ax = plt.subplots(figsize=(6.5, 8), facecolor='white')
        self.app.fig.patch.set_facecolor('white')
        
        self.app.canvas = FigureCanvasTkAgg(self.app.fig, master=container)
        self.app.canvas.draw()
        self.app.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def setup_initial_plot(self):
        """設定初始化圖表"""
        self.app.ax.clear()
        self.app.ax.set_facecolor('#fafafa')
        self.app.ax.set_title('SMA Area Trend Analysis\n(Updates every 15 seconds)', 
                         fontsize=12, fontweight='bold', color='#2c3e50', pad=20)
        self.app.ax.set_xlabel('Time (seconds)', fontsize=10, color='#34495e')
        self.app.ax.set_ylabel('Area (px²)', fontsize=10, color='#34495e')
        self.app.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # 添加等待數據的提示
        self.app.ax.text(0.5, 0.5, '⏳ Waiting for detection data...', 
                    transform=self.app.ax.transAxes, ha='center', va='center', 
                    fontsize=14, alpha=0.6, color='#95a5a6',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#bdc3c7', alpha=0.8))
        
        plt.tight_layout()
        self.app.canvas.draw()
    
    def setup_status_dashboard(self, parent):
        """設定底部狀態儀表板"""
        dashboard_frame = ttk.LabelFrame(parent, text="📋 系統狀態儀表板", style='Title.TLabelframe', padding="10")
        dashboard_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        dashboard_frame.columnconfigure(1, weight=1)
        
        # 左側狀態資訊
        status_left = ttk.Frame(dashboard_frame)
        status_left.grid(row=0, column=0, sticky=(tk.W, tk.N))
        
        self.app.status_var = tk.StringVar(value=f"🟢 系統就緒 ({'GPU' if self.app.gpu_available else 'CPU'} 模式)")
        status_label = ttk.Label(status_left, textvariable=self.app.status_var, 
                                font=('Microsoft JhengHei UI', 10, 'bold'), foreground='#27ae60')
        status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        # 中間性能指標
        perf_frame = ttk.Frame(dashboard_frame)
        perf_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        metrics_frame = ttk.Frame(perf_frame)
        metrics_frame.pack(anchor=tk.W)
        
        self.app.fps_var = tk.StringVar(value="🎯 FPS: 0")
        fps_label = ttk.Label(metrics_frame, textvariable=self.app.fps_var, 
                             font=('Microsoft JhengHei UI', 9), foreground='#3498db')
        fps_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        
        self.app.runtime_var = tk.StringVar(value="⏱️ Runtime: 00:00")
        runtime_label = ttk.Label(metrics_frame, textvariable=self.app.runtime_var, 
                                 font=('Microsoft JhengHei UI', 9), foreground='#9b59b6')
        runtime_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        
        # 面積資訊區域
        area_dashboard = ttk.LabelFrame(dashboard_frame, text="📏 即時面積資訊", padding="8")
        area_dashboard.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 0))
        area_dashboard.columnconfigure(0, weight=1)
        area_dashboard.columnconfigure(1, weight=1)
        area_dashboard.columnconfigure(2, weight=1)
        
        self.app.current_area_var = tk.StringVar(value="📐 當前面積: --")
        current_label = ttk.Label(area_dashboard, textvariable=self.app.current_area_var, 
                                 font=('Microsoft JhengHei UI', 9))
        current_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.app.avg_area_var = tk.StringVar(value="📊 平均面積: --")
        avg_label = ttk.Label(area_dashboard, textvariable=self.app.avg_area_var, 
                             font=('Microsoft JhengHei UI', 9))
        avg_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        self.app.sma_area_var = tk.StringVar(value="📈 SMA面積: --")
        sma_label = ttk.Label(area_dashboard, textvariable=self.app.sma_area_var, 
                             font=('Microsoft JhengHei UI', 10, 'bold'), foreground='#e74c3c')
        sma_label.grid(row=0, column=2, sticky=tk.W, padx=5)

if __name__ == "__main__":
    print("UIManager 類別已成功定義和載入")
    print("可用方法:", [method for method in dir(UIManager) if not method.startswith('_')])