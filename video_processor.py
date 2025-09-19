import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import threading
import time
import torch

class VideoProcessor:
    def __init__(self, app):
        self.app = app

    def process_video(self):
        """處理影像的主循環，支持預覽和推論模式（基本GPU支援）"""
        fps_counter = 0
        fps_start_time = time.time()
        consecutive_failures = 0
        max_failures = 10
        
        conf_thres = self.app.confidence_var.get()
        nan_gap = self.app.nan_gap_var.get()
        
        while (self.app.is_inference or self.app.is_preview) and self.app.cap and self.app.cap.isOpened():
            ret, frame = self.app.cap.read()
            
            if not ret:
                consecutive_failures += 1
                if self.app.stream_type == "rtsp" and consecutive_failures < max_failures:
                    time.sleep(0.5)
                    continue
                else:
                    break
            
            consecutive_failures = 0
            
            try:
                # 更新運行時間 (僅推論模式)
                if self.app.is_inference and self.app.inference_start_time:
                    runtime = int(time.time() - self.app.inference_start_time)
                    minutes, seconds = divmod(runtime, 60)
                    self.app.root.after(0, lambda: self.app.runtime_var.set(f"⏱️ Runtime: {minutes:02d}:{seconds:02d}"))
                
                # 使用實際經過時間作為時間戳
                time_sec = time.time() - (self.app.inference_start_time if self.app.is_inference else time.time())
                
                annotated_frame = frame.copy()
                detected = False
                current_areas = []
                
                # 儲存當前幀以供監控使用
                self.app.current_frame = frame.copy()
                
                if self.app.is_inference:
                    # YOLO推論（使用GPU）
                    with torch.no_grad():  # 節省GPU記憶體
                        results = self.app.model.predict(
                            frame, 
                            verbose=False, 
                            conf=conf_thres,
                            device=self.app.device,  # 指定使用GPU
                            half=self.app.gpu_available  # 如果有GPU則使用半精度運算
                        )[0]
                    
                    self.app.frame_group.append(results.boxes)
                    self.app.frame_times.append(time_sec)
                    
                    # 處理檢測結果並計算面積
                    if results.boxes is not None and len(results.boxes) > 0:
                        for box in results.boxes:
                            conf = box.conf.item()
                            cls_id = int(box.cls.item())
                            if cls_id == 0 and conf > conf_thres:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                area = (x2 - x1) * (y2 - y1)
                                current_areas.append(area)
                                detected = True
                    
                    # 記錄原始面積數據
                    if detected:
                        total_area = sum(current_areas)
                        self.app.raw_areas.append(total_area)
                        self.app.raw_time_stamps.append(time_sec)
                        self.app.recent_areas.append(total_area)
                        self.app.recent_times.append(time_sec)
                        self.app.last_detect_time = time.time()
                        
                        # 更新UI顯示
                        self.app.root.after(0, lambda: self.app.current_area_var.set(f"📐 當前面積: {total_area:.0f} px²"))
                    elif self.app.last_detect_time is not None and time_sec - (self.app.last_detect_time - self.app.inference_start_time) > nan_gap:
                        self.app.raw_areas.append(np.nan)
                        self.app.raw_time_stamps.append(time_sec)
                        self.app.recent_areas.append(np.nan)
                        self.app.recent_times.append(time_sec)
                    
                    # 每2幀計算平均面積
                    if len(self.app.frame_group) == 2:
                        group_time = np.mean(self.app.frame_times) if self.app.frame_times else time_sec
                        self.calculate_average_area_with_sma(group_time, conf_thres, nan_gap)
                        self.app.frame_group = []
                        self.app.frame_times = []
                    
                    # 繪製檢測結果
                    annotated_frame = results.plot()
                    
                    # 在影像上顯示面積資訊
                    if detected:
                        # 主要面積資訊背景
                        cv2.rectangle(annotated_frame, (8, 8), (400, 90), (0, 0, 0), -1)  # 黑色背景
                        cv2.rectangle(annotated_frame, (8, 8), (400, 90), (0, 255, 255), 2)  # 黃色邊框
                        
                        cv2.putText(annotated_frame, f"Total Area: {sum(current_areas):.0f} px", 
                                  (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # SMA面積資訊
                        if len(self.app.sma_areas) > 0:
                            latest_sma = self.app.sma_areas[-1]
                            if not np.isnan(latest_sma):
                                cv2.putText(annotated_frame, f"SMA Area: {latest_sma:.0f} px", 
                                          (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                        
                        # 檢測數量和信心度
                        cv2.putText(annotated_frame, f"Objects: {len(current_areas)} | Conf: {conf_thres:.2f}", 
                                  (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # GPU狀態顯示
                    if self.app.gpu_available:
                        cv2.putText(annotated_frame, f"GPU Mode - Device: {self.app.device}", 
                                  (15, annotated_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 儲存輸出影片
                    if self.app.output_video_writer:
                        self.app.output_video_writer.write(annotated_frame)
                else:
                    # 預覽模式，添加文字提示
                    device_text = f"Preview Mode (No Inference) - {('GPU' if self.app.gpu_available else 'CPU')} Ready"
                    cv2.putText(annotated_frame, device_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 轉換並顯示影像
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # 動態調整顯示尺寸
                label_width = self.app.image_label.winfo_width()
                label_height = self.app.image_label.winfo_height()
                
                if label_width > 100 and label_height > 100:
                    img_ratio = pil_image.width / pil_image.height
                    container_ratio = label_width / label_height
                    
                    if container_ratio > img_ratio:
                        new_height = min(label_height - 20, 600)
                        new_width = int(new_height * img_ratio)
                    else:
                        new_width = min(label_width - 20, 800)
                        new_height = int(new_width / img_ratio)
                    
                    display_size = (max(new_width, 480), max(new_height, 360))
                else:
                    display_size = (640, 480)
                
                pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                self.app.root.after(0, self.update_image_display, tk_image)
                
                # 計算FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    self.app.root.after(0, lambda: self.app.fps_var.set(f"🎯 FPS: {fps:.1f}"))
                    fps_counter = 0
                    fps_start_time = time.time()
                
                self.app.frame_count += 1
                
                # 調整處理速度
                time.sleep(1 / self.app.inference_fps_var.get())
                
                # 定期清理GPU記憶體
                if self.app.gpu_available and self.app.frame_count % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"處理影格時發生錯誤: {e}")
                self.app.root.after(0, lambda: self.app.system_info_var.set(f"❌ 處理錯誤: {str(e)[:30]}..."))
                break
        
        # 清理資源
        self.app.stop_processing()

    def calculate_average_area_with_sma(self, group_time, conf_thres, nan_gap):
        """計算2幀的平均面積並應用SMA平滑處理"""
        total_area, count = 0, 0
        
        for boxes in self.app.frame_group:
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                if cls_id == 0 and conf > conf_thres:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    total_area += area
                    count += 1
        
        if count > 0:
            avg_area = total_area / count
            self.app.avg_areas.append(avg_area)
            self.app.avg_time_stamps.append(group_time)
            self.app.last_detect_time = time.time()
            
            # 應用SMA平滑處理
            if len(self.app.avg_areas) >= 1:
                window_size = self.app.sma_window_var.get()
                # 使用pandas進行移動平均平滑
                smoothed = pd.Series(self.app.avg_areas).rolling(window=window_size, min_periods=1).mean()
                self.app.sma_areas = smoothed.tolist()
                self.app.sma_time_stamps = self.app.avg_time_stamps.copy()
                
                # 更新UI顯示
                latest_sma = self.app.sma_areas[-1]
                overall_avg = np.nanmean(self.app.avg_areas) if self.app.avg_areas else 0
                self.app.root.after(0, lambda: self.app.avg_area_var.set(f"📊 平均面積: {overall_avg:.0f} px²"))
                self.app.root.after(0, lambda: self.app.sma_area_var.set(f"📈 SMA面積: {latest_sma:.0f} px²"))
            
            # 如果監控啟動，立即檢查NG並處理警報
            if self.app.is_monitoring:
                self.app.root.after(0, self.app.check_and_handle_ng)
        else:
            if self.app.last_detect_time is not None and group_time - (self.app.last_detect_time - self.app.inference_start_time) > nan_gap:
                self.app.avg_areas.append(np.nan)
                self.app.avg_time_stamps.append(group_time)

    def update_image_display(self, tk_image):
        """更新影像顯示"""
        self.app.image_label.configure(image=tk_image, text="")
        self.app.image_label.image = tk_image  # 保持引用避免被垃圾回收