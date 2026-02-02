# Project Introduction

This project provides tools for analyzing chromatogram data using CNN and YOLO models along with a graphical user interface.

- `gui_peak_detection_CNN_YOLO.py`: Provides a graphical user interface for peak detection and segmentation of chromatogram data, supporting CNN and YOLO models.
- `read_cdf.py`: Used for reading and processing chromatogram data files in CDF format.

## Dependencies

This project requires the following Python packages:

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

Please ensure the latest versions of these packages are installed.

## Classes and Methods

### `EnhancedChromatogramGUI`

This class provides a graphical user interface for peak detection and segmentation of chromatogram data.

#### Main Methods

- `detect_peaks_and_contours()`: Detect peaks and contours.
- `display_roi_segmentation_detail()`: Display ROI segmentation details.
- `load_cdf_file()`: Load CDF file.
- `load_cnn_model()`: Load CNN model.
- `load_yolo_model()`: Load YOLO model.
- `perform_yolo_segmentation()`: Perform YOLO segmentation.
- `apply_models()`: Apply models.

### `ReadCdf`

This class is used for reading and processing chromatogram data files in CDF format.

#### Methods

- `drift_time()`: Calculate drift time.
- `interp()`: Interpolation processing.
- `modulation_time()`: Get modulation time.
- `scan_duration()`: Calculate scan duration.

## Usage

### GUI Interface

To use the graphical user interface, run the `gui_peak_detection_CNN_YOLO.py` script. The interface will allow you to load CDF files, select model weights, adjust parameters, and view detection results.

![Load CDF file](.\img\加载cdf文件.png)

Click on peak detection for preliminary contour detection and to construct the contour heat map (ROIs):

![Construct contour heat map (ROIs)](.\img\构建轮廓热图.png)

Select the classification model weights and YOLO segmentation model weights to automatically perform contour classification and segmentation. The results are as follows:

![Final result](.\img\最终结果.png)