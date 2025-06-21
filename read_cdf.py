import os
import netCDF4 as nc
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
import json



class ReadCdf:
    """
    用于读取和处理netCDF格式的色谱数据。
    """

    def __init__(self, filename):
        """
        初始化ReadCdf对象。

        :param filename: netCDF文件的路径。
        """
        self.filename = os.path.splitext(filename)[0]
        self.cdf = nc.Dataset(filename)
        self.df = self._load_data()

    def _load_data(self):
        """
        从netCDF文件中加载数据，并返回一个包含数据的DataFrame。
        """
        data = {}
        point_count = self.cdf.variables['point_count'][:].squeeze()
        #
        # # 获取其他变量数据
        # for var_name in self.cdf.variables:
        #     if var_name in ['mass_values', 'intensity_values']:
        #         continue  # 跳过这两个变量，稍后处理
        #     data[var_name] = np.array(self.cdf.variables[var_name][:]).squeeze()

        # 处理 mass_values 和 intensity_values
        mass_values = np.array(self.cdf.variables['mass_values'][:]).squeeze()
        intensity_values = np.array(self.cdf.variables['intensity_values'][:]).squeeze()

        mass_values_grouped = self._group_by_point_count(mass_values, point_count)
        intensity_values_grouped = self._group_by_point_count(intensity_values, point_count)

        data['scan_acquisition_time'] = self.cdf.variables['scan_acquisition_time'][:]
        data['total_intensity'] = self.cdf.variables['total_intensity'][:]
        data['mass_values'] = mass_values_grouped
        data['intensity_values'] = intensity_values_grouped

        return pd.DataFrame(data)

    def _group_by_point_count(self, values, point_count):
        """
        根据点数对值进行分组。

        :param values: 要分组的值列表。
        :param point_count: 每组的点数列表。
        :return: 分组后的值列表。
        """
        grouped_values = []
        index = 0
        for count in point_count:
            grouped_values.append(values[index:index + count])
            index += count
        return grouped_values

    def save_variables_to_csv(self, csv_filename='variables.csv'):
        """
        将所有变量保存为CSV文件，其中'mass_values'和'intensity_values'被序列化为JSON字符串。

        :param csv_filename: 要保存的CSV文件名。
        """
        # 创建一个副本以避免修改原始DataFrame
        df_to_save = self.df.copy()

        # 序列化'mass_values'和'intensity_values'为JSON字符串
        df_to_save['mass_values'] = df_to_save['mass_values'].apply(lambda x: json.dumps(x.tolist()))
        df_to_save['intensity_values'] = df_to_save['intensity_values'].apply(lambda x: json.dumps(x.tolist()))

        # 保存为CSV
        df_to_save.to_csv(csv_filename, index=False)
        print(f"所有变量已保存到 {csv_filename}")

    def scan_duration(self):
        """
        计算并返回扫描持续时间。

        :return: 扫描持续时间（浮点数）。
        """
        scan_times = self.df['scan_acquisition_time']
        if len(scan_times) > 1:
            return round(scan_times.iloc[1] - scan_times.iloc[0], 2)
        else:
            return 0

    def modulation_time(self, threshold = 1e5, window_size=100):
        """
        分析强度值中的峰值，返回峰值之间最常见的时间差。

        :param threshold: 识别峰值的强度阈值。
        :param window_size: 峰值检测的窗口大小。
        :return: 峰值之间最常见的时间差。
        """
        # 获取时间和强度数组
        time_array = self.df['scan_acquisition_time'].values
        value_array = self.df['total_intensity'].values

        # 找到最大值及其索引
        max_value = np.max(value_array)
        threshold = max_value * 0.1
        max_index = np.argmax(value_array)

        # 在最大值附近选择数据窗口
        start_index = max(0, max_index - 5000)
        end_index = min(len(value_array), max_index + 5000)
        window_data = value_array[start_index:end_index]
        window_time = time_array[start_index:end_index]

        # 在窗口中找到峰值
        peak_indices = self._find_peak_indices(window_data, threshold, window_size)

        # 计算相邻峰值之间的时间差
        if len(peak_indices) > 1:
            peak_time_differences = [
                window_time[peak_indices[i]] - window_time[peak_indices[i - 1]]
                for i in range(1, len(peak_indices))
            ]

            rounded_differences = [int(round(diff)) for diff in peak_time_differences]
            difference_counts = Counter(rounded_differences)

            # 找到最常见的时间差
            most_frequent_diff = difference_counts.most_common(1)[0][0]

            return int(most_frequent_diff * self.scan_duration() * 100)
        else:
            print("未找到足够的峰值。")
            return None

    def _find_peak_indices(self, data, threshold, window_size):
        """
        在数据中找到峰值索引。

        :param data: 要搜索峰值的数据数组。
        :param threshold: 识别峰值的强度阈值。
        :param window_size: 峰值检测的窗口大小。
        :return: 峰值索引列表。
        """
        peaks = []
        for i in range(0, len(data), window_size):
            window_end = min(i + window_size, len(data))
            window = data[i:window_end]
            if len(window) > 0:
                local_max = np.max(window)
                local_max_index = np.argmax(window) + i
                if local_max > threshold:
                    if (local_max_index == 0 or data[local_max_index] > data[local_max_index - 1]) and \
                       (local_max_index == len(data) - 1 or data[local_max_index] > data[local_max_index + 1]):
                        peaks.append(local_max_index)
        return peaks

    def drift_time(self):
        """
        计算并返回漂移时间。

        :return: 漂移时间（整数）。
        """
        height = max(self.df['total_intensity'][:50000])
        peaks_index, _ = find_peaks(self.df['total_intensity'][:50000],
                                    height=height, width=5)
        mod_time = self.modulation_time()
        scan_dur = self.scan_duration()

        if mod_time and scan_dur and len(peaks_index) > 0:
            drift = int((peaks_index[0] % (mod_time // scan_dur)) * 0.7)
            return drift  # 返回实际计算值
        else:
            return 0

    def interp(self, save_csv=False, save_heatmap=True, mod_time=None, drift_time=None):
        """
        对数据进行插值，保存热图，并返回矩阵。

        :param save_csv: 是否保存插值后的数据为CSV文件。
        :param save_heatmap: 是否保存热图图像。
        :return: 包含插值数据和矩阵的元组。
        """
        time_array = self.df['scan_acquisition_time'].values
        data_array = self.df['total_intensity'].values

        modulation_time_val = mod_time if mod_time is not None else self.modulation_time()
        drift_time_val = drift_time if drift_time is not None else self.drift_time()
        scan_duration = self.scan_duration()
        point_per_sec = int(1 / scan_duration)

        if not modulation_time_val or not scan_duration:
            print("由于缺少调制时间或扫描持续时间，无法进行插值。")
            return None

        # 创建统一的时间数组
        time_uniform = np.linspace(time_array[0], time_array[-1], int((time_array[-1] - time_array[0]) * point_per_sec))
        # 对强度值进行插值
        intensity_uniform = interp1d(time_array, data_array, kind='linear', fill_value="extrapolate")(time_uniform)

        # 应用漂移校正
        intensity_uniform = np.roll(intensity_uniform, drift_time_val)

        # 计算每个调制周期的点数
        num_points_per_modtime = int(modulation_time_val * point_per_sec)

        # 调整数组长度
        total_points = len(intensity_uniform)

        remainder = total_points % num_points_per_modtime
        if remainder != 0:
            intensity_uniform = intensity_uniform[:-remainder]
            time_uniform = time_uniform[:-remainder]

        # 将数据重塑为矩阵
        result_matrix = np.rot90(intensity_uniform.reshape(-1, num_points_per_modtime))
        time_matrix = np.rot90(time_uniform.reshape(-1, num_points_per_modtime))

        if save_csv:
            csv_filename = f'{self.filename}_interp.csv'
            np.savetxt(csv_filename, result_matrix, delimiter=',')
            print(f'插值数据已保存到 {csv_filename}')

        if save_heatmap:
            self._save_heatmap_white_background(result_matrix)
            self._save_heatmap(result_matrix)

        return intensity_uniform, result_matrix, time_uniform, time_matrix

    def _save_heatmap_white_background(self, matrix):
        """
        保存矩阵的热图为PNG文件，使用自定义颜色映射（白->蓝->绿->黄->红）。

        :param matrix: 要可视化为热图的矩阵。
        """
        # 创建自定义颜色映射
        colors = ['white', 'blue', 'green', 'green', 'yellow', 'red', 'black']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

        plt.figure(figsize=(69, 23))
        # 使用自定义颜色映射
        plt.imshow(matrix, cmap=cmap, aspect='auto')
        plt.axis('off')  # 移除坐标轴

        # 设置字体
        plt.rc("font", family='KaiTi')

        # 保存图像
        heatmap_filename = f"{self.filename}_wb.jpg"
        plt.savefig(heatmap_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _save_heatmap(self, matrix):
        """
        保存矩阵的热图为PNG文件。

        :param matrix: 要可视化为热图的矩阵。
        """
        plt.figure(figsize=(69, 23))
        plt.imshow(matrix, cmap='jet', aspect='auto')
        plt.axis('off')
        plt.rc("font", family='KaiTi')
        heatmap_filename = f"{self.filename}.jpg"
        plt.savefig(heatmap_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_interp_and_variables_csv(self, interp_csv='interp.csv', variables_csv='variables.csv'):
        """
        保存插值数据和所有变量到CSV文件。

        :param interp_csv: 插值矩阵保存的CSV文件名。
        :param variables_csv: 所有变量保存的CSV文件名。
        """
        # 保存插值数据
        result = self.interp(save_csv=True, save_heatmap=False)
        if result:
            intensity_uniform, result_matrix, time_uniform, time_matrix = result

        # 保存所有变量
        self.save_variables_to_csv(csv_filename=variables_csv)


if __name__ == '__main__':
    # 示例用法
    filename = 'data/21.CDF'
    chromatogram = ReadCdf(filename)
    print(f"文件名: {chromatogram.filename}")
    print(f"CDF 元数据: {chromatogram.cdf}")
    print(f"数据表首部:\n{chromatogram.df.head()}")
    print(f"调制时间: {chromatogram.modulation_time()}")
    print(f"漂移时间: {chromatogram.drift_time()}")
    print(f"扫描周期: {chromatogram.scan_duration()}")
    print(f"{chromatogram.interp()[1].shape}")

    # 保存插值数据和所有变量到CSV
    chromatogram.save_interp_and_variables_csv(variables_csv=f'{chromatogram.filename}_variables.csv')
