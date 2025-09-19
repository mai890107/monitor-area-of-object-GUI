import numpy as np
import pandas as pd
import time
class DataProcessor:
    @staticmethod
    def clean_data(areas, timestamps, gap_limit=3):
        """
        清理缺失值
        - gap_limit: 可補值的連續缺失數量 (例如 3 表示 ≤3 個缺失可線性補值)
        """
        if len(areas) == 0 or len(timestamps) == 0 or len(areas) != len(timestamps):
            return areas

        s = pd.Series(areas, index=timestamps)
        s.index = pd.to_timedelta(s.index, unit='s')
        areas_interpolated = s.interpolate(method="time", limit=gap_limit, limit_direction="both")
        areas_filled = areas_interpolated.fillna(0)  # 長缺口補 0

        return areas_filled.values

    @staticmethod
    def check_area_trend_timebased(areas, timestamps, minutes, epsilon, overlap_ratio=0.7, cooldown=180):
        """
        檢查區域數值在指定時間區間內的趨勢變化。
        流程:
        1.先把資料切成一段一段的時間區間(例如 3 分鐘或 5 分鐘)。
        2.算出「前一段」和「當前這一段」的平均值。
        3.比較兩段的平均值差多少:
        4.如果差異太小(幾乎沒變化),就當作是 「NG」 (平坦、異常)。
        5.否則就當作是「正常」。

        加入 cooldown 機制:如果某個時間點被判定為 NG,
        則必須間隔 cooldown 個 timestamp 之後,才會再次檢查 NG。

        參數:
            areas (list or numpy.ndarray): 區域數值的列表或陣列。
            timestamps (list or numpy.ndarray): 與區域數值對應的時間戳記 (單位: 秒)。
            minutes (int): 檢查的時間區間長度 (單位: 分鐘)。
            epsilon (float): 判斷平坦狀態的允許變化範圍 (例如 0.05 表示 5%)。
            overlap_ratio (float, 選填): 區間重疊比例,預設為 0.8。
        回傳:
            list: 包含每個時間區間檢查結果的字典列表,每個字典包含以下資訊:
                - "time_sec": 當前時間區間的結束時間 (秒)。
                - "minutes": 區間長度 (分鐘)。
                - "prev_avg_area": 前一區間的平均值。
                - "curr_avg_area": 當前區間的平均值。
                - "ratio": 當前區間與前一區間的平均值比率。
                - "status": 狀態 ("NG" 或 "NORMAL")。
        """
        if len(areas) == 0 or len(timestamps) == 0:
            return []
        L_sec = minutes * 60
        step_size = int(L_sec * (1 - overlap_ratio))  # 步進大小

        results = []
        timestamps = np.array(timestamps)
        areas = np.array(areas)

        last_ng_time = -np.inf  # 記錄上一次 NG 的時間

        for t_now in range(int(timestamps[0] + L_sec), int(timestamps[-1] + 1), step_size):
            t_prev = t_now - L_sec

            # 分割前後區間
            prev_window = areas[(timestamps >= t_prev - L_sec) & (timestamps < t_prev)]
            curr_window = areas[(timestamps >= t_prev) & (timestamps < t_now)]

            if len(prev_window) == 0 or len(curr_window) == 0:
                continue

            prev_avg = np.nanmean(prev_window)
            curr_avg = np.nanmean(curr_window)

            if prev_avg > 0:
                ratio = curr_avg / prev_avg
                status = "NG" if abs(ratio - 1) < epsilon else "NORMAL"

                # ✅ 加入 cooldown 限制
                if status == "NG" and (t_now - last_ng_time) <= cooldown:
                    status = "SKIP"  # 跳過,不重複標記
                if status == "NG":
                    last_ng_time = t_now  # 更新 NG 時間

                results.append({
                    "time_sec": t_now,
                    "minutes": minutes,
                    "prev_avg_area": prev_avg,
                    "curr_avg_area": curr_avg,
                    "ratio": ratio,
                    "status": status
                })
        return results

    @staticmethod
    def extract_ng_markers(results):
        """回傳所有 NG 狀態的時間戳記"""
        return [r["time_sec"] for r in results if r["status"] == "NG"]