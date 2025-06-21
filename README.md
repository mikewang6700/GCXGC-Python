# 项目简介

本项目提供了使用CNN、YOLO模型和图形用户界面分析色谱图数据的工具。

- `gui_peak_detection_CNN_YOLO.py`：提供一个图形用户界面，用于色谱图数据的峰检测和分割，支持 CNN 和 YOLO 模型。
- `read_cdf.py`：用于读取和处理 CDF 格式的色谱数据文件。

## 依赖包

本项目需要以下 Python 包：

- `os`
- `netCDF4`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `collections`
- `json`
- `tkinter`
- `PIL`
- `torch`
- `torchvision`
- `cv2`
- `ultralytics`

请确保安装了这些包的最新版本。

## 类和方法

### `EnhancedChromatogramGUI`

该类提供了一个图形用户界面，用于色谱图数据的峰检测和分割。

#### 主要方法


- `detect_peaks_and_contours(self)`: 检测峰和轮廓。
- `display_roi_segmentation_detail(self, riinfo_entry)`: 显示 ROI 分割详情。
- `load_cdf_file(self)`: 加载 CDF 文件。
- `load_cnn_model(self, weight_path)`: 加载 CNN 模型。
- `load_yolo_model(self, weight_path)`: 加载 YOLO 模型。
- `perform_yolo_segmentation(self, mask_info_entry)`: 执行 YOLO 分割。
- `apply_models(self, mask_data)`: 应用模型。

### `ReadCdf`

该类用于读取和处理 CDF 格式的色谱数据文件。

#### 方法

- `drift_time(self)`: 计算漂移时间。
- `interp(self, save_csv=False, saveheatmap=True, mod_time=None, drifttime=None)`: 插值处理。
- `modulation_time(self, threshold=1e5, window_size=100)`: 获取调制时间。
- `scan_duration(self)`: 获取扫描时间。

## 使用方法

### GUI 界面

要使用图形用户界面，请运行 `gui_peak_detection_CNN_YOLO.py` 脚本。界面将允许您加载 CDF 文件、选择模型权重、调整参数并查看检测结果。


