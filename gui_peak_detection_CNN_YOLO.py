import os
import gc
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sympy import gamma
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
from read_cdf import ReadCdf
import cv2
from ultralytics import YOLO

plt.rc('font', family='Microsoft YaHei')

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = x.view(-1, 256 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnhancedChromatogramGUI:
    def __init__(self, master):
        self.scan_duration = 0.01
        self.point_per_sec = int(1 // self.scan_duration)
        self.master = master
        self.master.title("GC × GC 色谱分析系统")
        self._setup_ui()
        self._init_vars()
        self.cnn_model = None
        self.yolo_model = None
        self.cnn_weight_path = tk.StringVar()
        self.yolo_weight_path = tk.StringVar()

    def _setup_ui(self):
        self._create_parameter_panel()

    def _init_vars(self):
        self.read_cdf = None
        self.current_file_path = None
        self.result_matrix = None
        self.heatmap_img = None  
        self.original_opencv_heatmap_img = None
        self.heatmapwb_img = None  
        self.current_column_idx = 0
        self.heatmap_ax = None 
        self.fig_canvas = None 
        self.clickable_rois_info_for_event = [] 

        self.instance_colors = [
            (255, 59, 59),  
            (59, 255, 59),  
            (59, 59, 255),  
            (255, 255, 59),  
            (255, 59, 255),  
            (59, 255, 255),  
            (255, 159, 59),  
            (159, 59, 255),  
            (59, 159, 255),  
            (128, 0, 0),  
            (0, 128, 0),  
            (0, 0, 128),  
            (128, 128, 0),  
            (128, 0, 128),  
            (0, 128, 128),  
        ]

    def _create_parameter_panel(self):
        param_frame = tk.Frame(self.master)
        param_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(param_frame, text="调制时间(s):").pack(side=tk.LEFT, padx=5)
        self.mod_time_entry = tk.Entry(param_frame, width=8)
        self.mod_time_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(param_frame, text="漂移时间(s):").pack(side=tk.LEFT, padx=5)
        self.drift_time_entry = tk.Entry(param_frame, width=8)
        self.drift_time_entry.pack(side=tk.LEFT, padx=5)

        self.update_btn = tk.Button(param_frame, text="更新参数", command=self.update_parameters)
        self.update_btn.pack(side=tk.LEFT, padx=10)

        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="打开 CDF 文件", command=self.load_cdf_file)
        filemenu.add_separator()
        filemenu.add_command(label="退出", command=self.master.quit)
        menubar.add_cascade(label="文件", menu=filemenu)
        self.master.config(menu=menubar)

        btn_frame = tk.Frame(self.master, height=40)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, anchor='s')

        self.prev_btn = tk.Button(btn_frame, text="上一列", command=self.show_previous_column)
        self.prev_btn.pack(side=tk.LEFT, padx=10, pady=5)

        self.next_btn = tk.Button(btn_frame, text="下一列", command=self.show_next_column)
        self.next_btn.pack(side=tk.LEFT, padx=10, pady=5)

        self.smooth_btn = tk.Button(btn_frame, text="平滑", command=self.smooth_and_reconstruct)
        self.smooth_btn.pack(side=tk.LEFT, padx=10, pady=5)

        self.detect_btn = tk.Button(btn_frame, text="峰检测", command=self.detect_peaks_and_contours)
        self.detect_btn.pack(side=tk.LEFT, padx=10, pady=5)

        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax_heatmap = self.fig.add_subplot(211)
        self.ax_projection = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH)

        self.initial_xlim = None
        self.initial_ylim = None

    def load_cdf_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CDF files", "*.cdf"), ("All files", "*.*")])
        if file_path:
            try:
                self.current_file_path = file_path
                self.read_cdf = ReadCdf(file_path)
                self.filename = self.read_cdf.filename,
                print(f"加载文件: {self.read_cdf.filename}")

                self.mod_time_entry.delete(0, tk.END)
                self.mod_time_entry.insert(0, f"{self.read_cdf.modulation_time():.0f}")
                self.drift_time_entry.delete(0, tk.END)
                self.drift_time_entry.insert(0, f"{self.read_cdf.drift_time():.0f}")

                self.scan_duration = self.read_cdf.scan_duration()
                self.point_per_sec = int(1 // self.scan_duration)

                result = self.read_cdf.interp(save_csv=False, save_heatmap=True)
                if result:
                    intensity_uniform, result_matrix, time_uniform, time_matrix = result
                    self.intensity_uniform = intensity_uniform
                    self.result_matrix = result_matrix
                    self.time_matrix = time_matrix
                    self.time_uniform = time_uniform

                    self.plot_heatmap()
                    self.plot_projection()
                    self.canvas.draw()
                    messagebox.showinfo("成功", "CDF 文件已成功加载并可视化。")
            except Exception as e:
                messagebox.showerror("错误", f"加载 CDF 文件时出错:\n{e}")

    def select_file_path(self, string_var_to_update, file_types_list):
        
        filepath = filedialog.askopenfilename(
            parent=self.master, 
            title="请选择文件",
            filetypes=file_types_list,
            defaultextension=".pt" 
        )
        if filepath:  
            string_var_to_update.set(filepath)

    def update_parameters(self):
        try:
            new_mod_time = int(self.mod_time_entry.get())
            new_drift_time = int(self.drift_time_entry.get())

            result = self.read_cdf.interp(mod_time=new_mod_time, drift_time=new_drift_time, save_csv=False,
                                          save_heatmap=False)
            if result and result[1].shape[0] > 0:
                self.intensity_uniform, self.result_matrix, self.time_uniform, self.time_matrix = result
                if self.result_matrix is not None:
                    self.original_opencv_heatmap_img = self._save_heatmap(self.result_matrix.copy())

                self.plot_heatmap()
                self.plot_projection()
                self.canvas.draw()
            else:
                raise ValueError("生成矩阵无效")
        except ValueError as ve:
            messagebox.showerror("计算错误", f"参数格式错误: {str(ve)}")
        except Exception as e:
            messagebox.showerror("运行时错误", f"数据重塑失败: {str(e)}")

    def plot_heatmap(self):
        colors = ['white', 'blue', 'green', 'yellow', 'red', 'black']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        if self.result_matrix is not None:
            self.original_opencv_heatmap_img = self._save_heatmap(self.result_matrix.copy())
        result_matrix_display = np.flipud(self.result_matrix)
        modulation_time_sec = int(self.mod_time_entry.get())

        scan_acquisition_time_min = self.time_uniform / 60.0

        self.ax_heatmap.clear()
        self.ax_heatmap.imshow(
            result_matrix_display, cmap=cmap, aspect='auto', origin='lower',
            extent=[scan_acquisition_time_min[0], scan_acquisition_time_min[-1], 0, modulation_time_sec]
        )
        self.ax_heatmap.set_xlabel('Scan Acquisition Time (min)')
        self.ax_heatmap.set_ylabel('Modulation Periods (s)')
        self.ax_heatmap.set_title('二维图')

        self.initial_xlim = self.ax_heatmap.get_xlim()
        self.initial_ylim = self.ax_heatmap.get_ylim()

    def show_projection(self, column_idx):
        if self.result_matrix is None or self.time_matrix is None:
            return

        num_mod_periods, num_points_per_modtime = self.result_matrix.shape
        if column_idx < 0 or column_idx >= num_points_per_modtime:
            return

        self.current_column_idx = column_idx
        column_data = self.result_matrix[:, column_idx]
        time_column = self.time_matrix[:, column_idx] / 60.0

        self.ax_projection.clear()
        self.ax_projection.plot(time_column, column_data, color='blue')
        self.ax_projection.set_xlabel('Time (min)')
        self.ax_projection.set_ylabel('Intensity Projection')
        self.ax_projection.set_title(f'一维投影 (列 {column_idx + 1})')
        self.canvas.draw()

    def plot_projection(self):
        self.ax_projection.clear()
        projection = np.sum(self.result_matrix, axis=0)
        time_second_dimension = self.time_matrix[0, :] / 60.0
        self.ax_projection.plot(time_second_dimension, projection, color='blue')
        self.ax_projection.set_xlabel('Second Dimension Time (min)')
        self.ax_projection.set_ylabel('Intensity Projection')
        self.ax_projection.set_title('一维投影')
        self.canvas.draw()

    def detect_peaks_and_contours(self):
        try:
            
            if self.result_matrix is None:
                messagebox.showerror("错误", "请先加载CDF文件并生成热图数据。")
                return

            mod_time = int(self.mod_time_entry.get())
            drift_time = int(self.drift_time_entry.get())
            if mod_time <= 0 or drift_time <= 0:
                raise ValueError("调制时间和漂移时间必须大于0")

            
            gamma_corrected_cv_img, contours = self.process_heatmap_contours()
            if gamma_corrected_cv_img is None or contours is None:  
                messagebox.showerror("检测错误", "处理热图或提取轮廓失败。")
                return

            
            mask_data = self._draw_and_save_contours(gamma_corrected_cv_img, contours)
            if not mask_data:  
                messagebox.showinfo("提示", "未检测到有效轮廓或未能生成掩码。")
                

            self._show_detection_results(mask_data)
        except ValueError as ve:  
            messagebox.showerror("参数错误", f"{type(ve).__name__}: {str(ve)}")
        except Exception as e:
            messagebox.showerror("检测错误", f"发生意外错误: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()  

    def apply_gamma(self, matrix, gamma):
        max_val = np.max(matrix)
        if max_val == 0:
            return matrix.copy()
        normalized = matrix / max_val
        gamma_mapped = np.power(normalized, gamma)
        return gamma_mapped * max_val

    def _save_heatmap(self, matrix):
        matrix_norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(matrix_norm, cv2.COLORMAP_JET)
        return heatmap

    def _save_heatmap_grayscale(self, matrix):
        matrix_norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if len(matrix_norm.shape) == 2:  
            return matrix_norm
        elif len(matrix_norm.shape) == 3 and matrix_norm.shape[2] == 3:  
            return cv2.cvtColor(matrix_norm, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported matrix shape for grayscale conversion")

    def _save_heatmap_white_background(self, matrix):
        matrix_norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        lut = np.zeros((256, 3), dtype=np.uint8)
        colors = [
            (255, 255, 255),  
            (0, 0, 255),  
            (0, 255, 0),  
            (255, 255, 0),  
            (255, 0, 0),  
            (0, 0, 0)  
        ]
        for i in range(256):
            if i < 51:
                lut[i, :] = colors[0]
            elif i < 102:
                lut[i, :] = colors[1]
            elif i < 153:
                lut[i, :] = colors[2]
            elif i < 204:
                lut[i, :] = colors[3]
            else:
                lut[i, :] = colors[4]
        heatmap_wb = lut[matrix_norm]
        return heatmap_wb

    def process_heatmap_contours(self):
        if self.result_matrix is None:
            messagebox.showerror("错误", "请先加载数据再进行峰检测。")
            return None, None  

        gamma_val = 0.5  
        gamma_mapped_matrix = self.apply_gamma(self.result_matrix, gamma_val)

        img_gamma_corrected_cv = self._save_heatmap(gamma_mapped_matrix)
        img_gray_gamma_corrected = self._save_heatmap_grayscale(gamma_mapped_matrix)  

        self.heatmap_img = img_gamma_corrected_cv
        
        ret, binary = cv2.threshold(img_gray_gamma_corrected, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if not ret:  
            print("OTSU thresholding failed, using a default threshold.")
            _, binary = cv2.threshold(img_gray_gamma_corrected, 10, 255, cv2.THRESH_BINARY)

        laplacian = cv2.Laplacian(binary, cv2.CV_64F)
        laplacian = np.uint8(np.abs(laplacian))
        contours, _ = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"检测到 {len(contours)} 个轮廓")

        
        return img_gamma_corrected_cv, contours

    def _draw_and_save_contours(self, gamma_corrected_cv_img, contours):
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)

        mask_info_list = []  
        target_size = (224, 224)
        background_color = (0, 0, 0)

        filename_base = os.path.splitext(os.path.basename(self.current_file_path))[0]
        roi_base_heatmap_filename = f"{filename_base}_gamma_corrected_roi_base.jpg"
        if gamma_corrected_cv_img is not None:  
            cv2.imwrite(os.path.join(output_dir, roi_base_heatmap_filename), gamma_corrected_cv_img)
        else:
            roi_base_heatmap_filename = None  

        for idx, cnt in enumerate(contours):
            if len(cnt) < 3:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue

            try:
                
                mask = np.zeros_like(gamma_corrected_cv_img)
                cv2.drawContours(mask, [cnt], -1, (255, 255, 255), cv2.FILLED)
                colored_mask = cv2.bitwise_and(gamma_corrected_cv_img, mask)
                roi = colored_mask[y:y + h, x:x + w]

                scale = min(target_size[0] / w,
                            target_size[1] / h) if w > 0 and h > 0 else 1.0  
                new_w = int(w * scale)
                new_h = int(h * scale)

                if new_w <= 0 or new_h <= 0:  
                    print(f"Skipping contour {idx} due to zero/negative resized dimensions.")
                    continue

                resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

                padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                padded[:] = background_color

                
                y_offset_padding = (target_size[0] - new_h) // 2  
                x_offset_padding = (target_size[1] - new_w) // 2  

                padded[y_offset_padding:y_offset_padding + new_h, x_offset_padding:x_offset_padding + new_w] = resized

                mask_dir = os.path.join(output_dir, f"mask_{idx:04d}")
                os.makedirs(mask_dir, exist_ok=True)
                mask_path = os.path.join(mask_dir, "colored_mask.jpg")
                cv2.imwrite(mask_path, padded)

                mask_info_list.append({
                    "id": idx,
                    "mask_path": mask_path, 
                    "original_bbox": (x, y, w, h),
                    "scale_factor": scale,
                    "padding_offsets": (x_offset_padding, y_offset_padding),
                    "contour_points": cnt.tolist(),
                    "roi_base_heatmap": roi_base_heatmap_filename,
                    "classification": None,
                    "yolo_segmentation_contours_full_heatmap": [], 
                    "yolo_segmented_roi_path": None 
                })
                del mask, colored_mask, roi, resized, padded
                if idx % 10 == 0:
                    gc.collect()
            except MemoryError:
                print(f"内存不足，跳过轮廓 {idx}")
                continue
            except Exception as e:
                print(f"处理轮廓 {idx} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        print(f"成功生成 {len(mask_info_list)} 个彩色 mask")
        return mask_info_list

    def _transform_yolo_mask_to_full_heatmap_contours(self, yolo_mask_on_roi, roi_info):
        if yolo_mask_on_roi is None or yolo_mask_on_roi.sum() == 0:
            return []
        mask_uint8 = (yolo_mask_on_roi > 0).astype(np.uint8) * 255 

        contours_on_roi, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        transformed_contours = []
        original_bbox_x, original_bbox_y, _, _ = roi_info["original_bbox"]
        scale = roi_info["scale_factor"]
        pad_x_offset, pad_y_offset = roi_info["padding_offsets"]

        if scale == 0:  
            print("Warning: Scale factor is zero in _transform_yolo_mask_to_full_heatmap_contours.")
            return []

        for cnt_on_roi in contours_on_roi:
            if len(cnt_on_roi) == 0:
                continue
            
            cnt_on_roi_float = cnt_on_roi.astype(np.float32)

            cnt_unpadded = cnt_on_roi_float - np.array([[[pad_x_offset, pad_y_offset]]], dtype=np.float32)

            cnt_unscaled = cnt_unpadded / scale

            cnt_full_heatmap = cnt_unscaled + np.array([[[original_bbox_x, original_bbox_y]]], dtype=np.float32)

            transformed_contours.append(cnt_full_heatmap.astype(np.int32)) 

        return transformed_contours

    def _export_labelme_format(self, mask_data):
        labelme_data = {
            "version": "5.3.0",
            "flags": {},
            "shapes": [],
            "imagePath": f"{os.path.basename(self.current_file_path)}_heatmap_wb.png",
            "imageData": None,
            "imageHeight": self.result_matrix.shape[0],
            "imageWidth": self.result_matrix.shape[1]
        }

        for mask in mask_data:
            labelme_data["shapes"].append({
                "label": f"peak_{mask['id']}",
                "points": mask["contour_points"],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
        print(f"导出 {len(mask_data)} 个轮廓到 LabelMe JSON")

        save_path = os.path.join(os.path.dirname(self.current_file_path), "annotation.json")
        with open(save_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)
        messagebox.showinfo("导出成功", f"标注文件已保存到：{save_path}")

    def show_previous_column(self):
        if self.current_column_idx is not None:
            self.show_projection(self.current_column_idx - 1)

    def show_next_column(self):
        if self.current_column_idx is not None:
            self.show_projection(self.current_column_idx + 1)

    def smooth_and_reconstruct(self):
        drift_time = int(self.drift_time_entry.get())
        modulation_time = int(self.mod_time_entry.get())
        point_per_sec = int(self.point_per_sec)
        if self.intensity_uniform is None or self.result_matrix is None:
            messagebox.showerror("错误", "尚未加载数据，无法进行平滑处理。")
            return

        try:
            smoothed_data = gaussian_filter(self.intensity_uniform, sigma=5)
            intensity_uniform = np.where(smoothed_data < 0, 0, smoothed_data)
            num_points_per_modtime = int(modulation_time * point_per_sec)
            total_points = len(intensity_uniform)
            remainder = total_points % num_points_per_modtime
            if remainder != 0:
                intensity_uniform = intensity_uniform[:-remainder]
            self.result_matrix = np.rot90(intensity_uniform.reshape(-1, num_points_per_modtime))
            self.plot_heatmap()
            self.show_projection(self.current_column_idx)
            self.canvas.draw()
            messagebox.showinfo("成功", "平滑处理完成")
        except Exception as e:
            messagebox.showerror("错误", f"平滑处理时发生错误:\n{e}")

    def load_cnn_model(self, weight_path):
        
        if "cnn5" in weight_path.lower():
            self.cnn_model = CNN5()  
        elif "resnet18" in weight_path.lower():
            self.cnn_model = resnet18()
        else:
            raise ValueError("不支持的 CNN 模型权重文件")
        self.cnn_model.load_state_dict(torch.load(weight_path))
        self.cnn_model.eval()

    def classify_mask(self, mask_path):
        img = cv2.imread(mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.cnn_model(img_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def load_yolo_model(self, weight_path):
        self.yolo_model = YOLO(weight_path)
        yolo_model = YOLO(weight_path)
        print(f"YOLO分割模型加载成功 (任务类型: {yolo_model.task})。")
        if yolo_model.task != 'segment':
            print(f"警告：加载的YOLO模型任务类型为 '{yolo_model.task}'，而非 'segment'。分割可能无法按预期工作。")

    def perform_yolo_segmentation(self, mask_info_entry):
        if not self.yolo_model:
            messagebox.showerror("错误", "YOLO 模型未加载。")
            mask_info_entry["yolo_segmented_roi_path"] = None
            return

        yolo_input_image_path = mask_info_entry["mask_path"]
        if not os.path.exists(yolo_input_image_path):
            print(f"YOLO 输入图像未找到: {yolo_input_image_path}")
            mask_info_entry["yolo_segmented_roi_path"] = None
            return

        original_roi_image = cv2.imread(yolo_input_image_path)
        if original_roi_image is None:
            print(f"无法读取 YOLO 输入图像: {yolo_input_image_path}")
            mask_info_entry["yolo_segmented_roi_path"] = None
            return

        visualization_on_roi = original_roi_image.copy()

        target_height, target_width = visualization_on_roi.shape[:2]

        try:
            results = self.yolo_model.predict(source=original_roi_image, verbose=False, save=False)
        except Exception as e:
            print(f"YOLO 预测失败 ({mask_info_entry['id']}): {e}")
            mask_info_entry["yolo_segmented_roi_path"] = None
            return

        mask_info_entry["yolo_segmentation_contours_full_heatmap"] = []
        num_instances_detected = 0

        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            masks_tensor_list = results[0].masks.data

            for i, mask_tensor in enumerate(masks_tensor_list):

                mask_float_from_tensor = mask_tensor.cpu().numpy()

                if mask_float_from_tensor.sum() == 0:
                    continue

                num_instances_detected += 1
                color = self.instance_colors[i % len(self.instance_colors)]

                resized_mask_float = cv2.resize(mask_float_from_tensor,
                                                (target_width, target_height),
                                                interpolation=cv2.INTER_LINEAR)

                boolean_mask_for_indexing = (resized_mask_float > 0.5)

                visualization_on_roi[boolean_mask_for_indexing] = color

            if num_instances_detected > 0:
                print(f"Mask {mask_info_entry['id']}: YOLO 检测到 {num_instances_detected} 个实例。")
                mask_dir = os.path.dirname(mask_info_entry["mask_path"])
                yolo_viz_filename = f"roi_{mask_info_entry['id']}_yolo_segmented.jpg"
                yolo_viz_path = os.path.join(mask_dir, yolo_viz_filename)

                try:
                    cv2.imwrite(yolo_viz_path, visualization_on_roi)
                    mask_info_entry["yolo_segmented_roi_path"] = yolo_viz_path
                    print(f"已保存YOLO分割可视化图像至: {yolo_viz_path}")
                except Exception as e:
                    print(f"保存YOLO可视化图像失败: {e}")
                    mask_info_entry["yolo_segmented_roi_path"] = None
            else:
                print(f"YOLO 未在 mask {mask_info_entry['id']} 中检测到有效实例。")
                mask_info_entry["yolo_segmented_roi_path"] = None

        else:
            print(f"YOLO 未在 mask {mask_info_entry['id']} 中返回掩码数据或检测到分割结果。")
            mask_info_entry["yolo_segmented_roi_path"] = None

    def apply_models(self, mask_data):
        cnn_weight_path_str = self.cnn_weight_path.get()
        yolo_weight_path_str = self.yolo_weight_path.get()

        if not cnn_weight_path_str:
            messagebox.showerror("错误", "请先选择 CNN 模型权重。")
            return
        
        if not hasattr(self, 'loaded_cnn_weight_path') or self.loaded_cnn_weight_path != cnn_weight_path_str:
            try:
                self.load_cnn_model(cnn_weight_path_str)
                self.loaded_cnn_weight_path = cnn_weight_path_str
            except Exception as e:
                messagebox.showerror("错误", f"加载 CNN 模型失败: {e}")
                self.cnn_model = None 
                return

        needs_yolo = False 
        for mask_entry_check in mask_data:
            pass 

        if yolo_weight_path_str: 
             if not hasattr(self, 'loaded_yolo_weight_path') or self.loaded_yolo_weight_path != yolo_weight_path_str:
                try:
                    self.load_yolo_model(yolo_weight_path_str)
                    self.loaded_yolo_weight_path = yolo_weight_path_str
                except Exception as e:
                    messagebox.showerror("错误", f"加载 YOLO 模型失败: {e}")
                    self.yolo_model = None 
        else: 
            self.yolo_model = None 

        if not self.cnn_model:
            messagebox.showerror("错误", "CNN模型未能成功加载。")
            return

        print("开始应用模型...")
        actual_yolo_needed_and_missing = False
        for mask_entry in mask_data:
            try:
                classification_result = self.classify_mask(mask_entry["mask_path"])
                mask_entry["classification"] = classification_result
                print(f"Mask {mask_entry['id']} 分类为: {classification_result}")

                if classification_result == 0:
                    if self.yolo_model and self.yolo_weight_path.get(): 
                        print(f"对 Mask {mask_entry['id']} (分类为0) 执行 YOLO 分割并在ROI上可视化...")
                        self.perform_yolo_segmentation(mask_entry)
                    else:
                        print(f"跳过 Mask {mask_entry['id']} 的 YOLO 分割 (模型未加载或路径未指定)。")
                        mask_entry["yolo_segmented_roi_path"] = None
                        if not self.yolo_model and yolo_weight_path_str: 
                            actual_yolo_needed_and_missing = True
                        elif not yolo_weight_path_str: 
                             actual_yolo_needed_and_missing = True

            except Exception as e:
                print(f"处理 mask {mask_entry['id']} 时出错: {e}")
                import traceback
                traceback.print_exc()

        if actual_yolo_needed_and_missing and not yolo_weight_path_str:
             messagebox.showwarning("YOLO处理提示", "部分区域分类为0，但YOLO模型权重未选择，因此未进行分割。")
        elif actual_yolo_needed_and_missing and yolo_weight_path_str and not self.yolo_model:
             messagebox.showwarning("YOLO处理提示", "部分区域分类为0，YOLO权重已选择但模型加载失败，因此未进行分割。")


        print("模型应用完成。")
        messagebox.showinfo("模型应用完成",
                            "模型已应用。对于分类为0的区域，如果YOLO成功分割，"
                            "带有彩色分割的ROI图像已保存在各自的 'results/mask_xxxx/' 目录下 "
                            "(文件名为 roi_xxxx_yolo_segmented.jpg)。")
        
        self.display_results(mask_data, show_final_ml_results=True)
        

    def display_results(self, mask_data, show_segmentation=True, show_final_ml_results=False):
        print(f"\n--- display_results called: show_final_ml_results={show_final_ml_results} ---")
        print(f"Number of mask_data entries: {len(mask_data)}")
        if mask_data:
            sample_entry_for_print = mask_data[0]
            if show_final_ml_results:
                first_class_0 = next((m for m in mask_data if m.get("classification") == 0), None)
                if first_class_0:
                    sample_entry_for_print = first_class_0
            print(
                f"Sample mask_entry (ID {sample_entry_for_print.get('id')}) classification: {sample_entry_for_print.get('classification')}")

        
        if self.fig_canvas is None:
            print("[DEBUG Canvas] self.fig_canvas is None. Creating new canvas, figure, and connecting event.")
            for widget in self.heatmap_display_frame.winfo_children():
                widget.destroy()
            persistent_fig = plt.Figure(figsize=(10, 8))
            self.heatmap_ax = persistent_fig.add_subplot(111)
            self.fig_canvas = FigureCanvasTkAgg(persistent_fig, master=self.heatmap_display_frame)
            self.click_event_cid = self.fig_canvas.mpl_connect('button_press_event', self.on_heatmap_click)
            self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            print("[DEBUG Canvas] New canvas created, packed, and button_press_event connected.")
        else:
            print("[DEBUG Canvas] self.fig_canvas exists. Clearing its figure and re-adding axes.")
            self.fig_canvas.figure.clf()
            self.heatmap_ax = self.fig_canvas.figure.add_subplot(111)

        
        self.clickable_rois_info_for_event = []
        legend_elements = {}
        contours_drawn_count = 0
        target_fill_alpha = 0.4  

        if self.original_opencv_heatmap_img is not None:
            print(f"[DEBUG Draw] Drawing background image, shape: {self.original_opencv_heatmap_img.shape}")
            display_bg_img_rgb = cv2.cvtColor(self.original_opencv_heatmap_img, cv2.COLOR_BGR2RGB)
            self.heatmap_ax.imshow(display_bg_img_rgb)

            for mask_entry in mask_data:
                original_contour_on_full_heatmap = np.array(mask_entry["contour_points"])

                contour_points_for_draw = original_contour_on_full_heatmap.reshape(-1, 2)
                x_coords = contour_points_for_draw[:, 0]
                y_coords = contour_points_for_draw[:, 1]

                label_text = None
                artist_for_legend = None  

                if show_final_ml_results:
                    classification = mask_entry.get("classification")

                    should_fill_contour = False
                    face_color_for_fill = None  
                    edge_color_for_fill_or_line = None  

                    if classification == 1:
                        label_text = '单峰区域'
                        face_color_for_fill = 'green'
                        edge_color_for_fill_or_line = 'darkgreen'  
                        should_fill_contour = True
                    elif classification == 0:
                        label_text = '多峰区域（点击查看详情）'
                        face_color_for_fill = ''
                        edge_color_for_fill_or_line = 'darkorange'  
                        should_fill_contour = True

                        self.clickable_rois_info_for_event.append({
                            "id": mask_entry["id"],
                            "original_bbox": mask_entry["original_bbox"],  
                            "contour_points": contour_points_for_draw,  
                            "classification": classification,
                            "yolo_segmented_roi_path": mask_entry.get("yolo_segmented_roi_path")
                        })

                    if should_fill_contour and label_text and face_color_for_fill:
                        
                        patch = self.heatmap_ax.fill(x_coords, y_coords,
                                                     facecolor=face_color_for_fill,
                                                     alpha=target_fill_alpha,  
                                                     edgecolor=edge_color_for_fill_or_line,  
                                                     linewidth=1,  
                                                     label=label_text)  
                        artist_for_legend = patch[0]  
                    elif label_text and edge_color_for_fill_or_line:  
                        line, = self.heatmap_ax.plot(x_coords, y_coords,
                                                     color=edge_color_for_fill_or_line,
                                                     linewidth=1.5,  
                                                     label=label_text)
                        artist_for_legend = line

                else:  
                    label_text = 'Detected Peak Contour'
                    line_color = 'red'

                    line, = self.heatmap_ax.plot(x_coords, y_coords,
                                                 color=line_color,
                                                 linestyle='-',
                                                 linewidth=1,  
                                                 label=label_text)
                    artist_for_legend = line

                if artist_for_legend and label_text and label_text not in legend_elements:
                    legend_elements[label_text] = artist_for_legend

                contours_drawn_count += 1

            print(f"[DEBUG Draw] Number of contours/fills processed: {contours_drawn_count}")

            if legend_elements:
                self.heatmap_ax.legend(legend_elements.values(), legend_elements.keys(), loc='upper right',
                                       fontsize='small')

            self.heatmap_ax.axis('on')  
            title = "热图与模型结果" if show_final_ml_results else "热图与初始检测轮廓"
            self.heatmap_ax.set_title(title)
            print(f"[DEBUG Draw] Set title: {title}")

            if show_final_ml_results and self.clickable_rois_info_for_event:
                print(
                    f"\n[DEBUG ClickableROIs] Final self.clickable_rois_info_for_event (Total: {len(self.clickable_rois_info_for_event)}):")
                
            elif show_final_ml_results and not self.clickable_rois_info_for_event:
                print(
                    f"\n[DEBUG ClickableROIs] self.clickable_rois_info_for_event is EMPTY (show_final_ml_results was True).")

        else:
            print("[DEBUG Draw] No background image. Displaying text on heatmap_ax.")
            self.heatmap_ax.text(0.5, 0.5, "无可用背景热图显示", ha='center', va='center')

        
        try:
            self.fig_canvas.draw_idle()
            print("[DEBUG Canvas] fig_canvas.draw_idle() called successfully.")
        except Exception as e:
            print(f"[ERROR Canvas] Error during fig_canvas.draw_idle(): {e}")
            import traceback
            traceback.print_exc()

        print("--- display_results finished ---\n")

    def on_heatmap_click(self, event):
        print("\n--- [DEBUG] on_heatmap_click triggered ---")
        if event.inaxes != self.heatmap_ax:
            print("[DEBUG] Click was outside heatmap_ax. Returning.")
            return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            print("[DEBUG] Click coordinates (event.xdata, event.ydata) are None. Returning.")
            return

        print(f"[DEBUG] Heatmap clicked at (data coords): ({click_x:.2f}, {click_y:.2f})")

        if not hasattr(self, 'clickable_rois_info_for_event') or not self.clickable_rois_info_for_event:
            print("[DEBUG] self.clickable_rois_info_for_event is missing or empty. Returning.")
            return
        print(f"[DEBUG] Num items in self.clickable_rois_info_for_event: {len(self.clickable_rois_info_for_event)}")

        found_target_roi = None
        for idx, roi_info in enumerate(self.clickable_rois_info_for_event):
            x_roi, y_roi, w_roi, h_roi = roi_info["original_bbox"]

            if idx < 10 or (
                    click_x > x_roi and click_x < x_roi + w_roi and click_y > y_roi and click_y < y_roi + h_roi):  
                print(
                    f"  [DEBUG] Checking against ROI ID: {roi_info.get('id')}, BBox: ({x_roi:.1f},{y_roi:.1f},{w_roi:.1f},{h_roi:.1f})")

            if x_roi <= click_x < (x_roi + w_roi) and \
                    y_roi <= click_y < (y_roi + h_roi):
                found_target_roi = roi_info
                print(
                    f"  >>> [DEBUG] Click HIT ROI ID: {found_target_roi.get('id')}, BBox: {found_target_roi.get('original_bbox')}")
                break

        if found_target_roi:
            print(
                f"[DEBUG] Target ROI found. Calling display_roi_segmentation_detail for ROI ID: {found_target_roi.get('id')}")
            self.display_roi_segmentation_detail(found_target_roi)
        else:
            print("[DEBUG] No target ROI found for this click among clickable ROIs.")
            
            for widget in self.roi_detail_frame.winfo_children():
                widget.destroy()
            no_roi_label = tk.Label(self.roi_detail_frame,
                                    text="未选中有效的0类ROI区域，\n或所选区域无分割详情。",
                                    wraplength=300, justify=tk.CENTER)  
            no_roi_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        print("--- [DEBUG] on_heatmap_click finished ---")

    def display_roi_segmentation_detail(self, roi_info_entry):
        print(f"\n--- [DEBUG] display_roi_segmentation_detail called for ROI ID: {roi_info_entry.get('id')} ---")
        
        for widget in self.roi_detail_frame.winfo_children():
            widget.destroy()

        yolo_viz_path = roi_info_entry.get("yolo_segmented_roi_path")
        roi_id = roi_info_entry.get("id", "N/A")
        print(f"[DEBUG] YOLO segmented ROI path from entry: '{yolo_viz_path}'")

        if yolo_viz_path and isinstance(yolo_viz_path, str) and os.path.exists(yolo_viz_path):
            print(f"[DEBUG] Path exists. Attempting to load image: {yolo_viz_path}")
            try:
                img_pil = Image.open(yolo_viz_path)
                print(f"[DEBUG] Image loaded successfully using Pillow: size {img_pil.size}, mode {img_pil.mode}")

                self.roi_detail_frame.update_idletasks()  
                
                available_width = self.roi_detail_frame.winfo_width() - 20  
                available_height = self.roi_detail_frame.winfo_height() - 50  

                
                target_w = max(available_width, 100)
                target_h = max(available_height, 100)
                print(
                    f"[DEBUG] Target thumbnail size for detail frame: ({target_w}, {target_h}) based on available ({available_width},{available_height})")

                img_pil.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
                print(f"[DEBUG] Image thumbnailed to: {img_pil.size}")

                photo = ImageTk.PhotoImage(img_pil)
                print("[DEBUG] ImageTk.PhotoImage created.")

                img_label = tk.Label(self.roi_detail_frame, image=photo)
                img_label.image = photo  
                img_label.pack(pady=5)

                info_label_text = f"ROI ID: {roi_id} (YOLO分割结果)"
                tk.Label(self.roi_detail_frame, text=info_label_text).pack(pady=5)
                print(f"[DEBUG] Detail image and label packed for ROI ID: {roi_id}")

            except Exception as e:
                print(f"[ERROR] In display_roi_segmentation_detail while processing image: {e}")
                import traceback
                traceback.print_exc()
                
                error_text = f"无法加载ROI {roi_id} 的分割图像。\n路径: {os.path.basename(yolo_viz_path)}\n错误详情: {str(e)[:200]}"  
                error_label = tk.Label(self.roi_detail_frame,
                                       text=error_text,
                                       wraplength=max(200, self.roi_detail_frame.winfo_width() - 20),  
                                       fg="red", justify=tk.LEFT)
                error_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            error_message = f"ROI ID: {roi_id}\n"
            if not yolo_viz_path or not isinstance(yolo_viz_path, str):
                error_message += "无有效的YOLO分割图像路径信息。"
            elif not os.path.exists(yolo_viz_path):
                error_message += f"YOLO分割图像文件未找到:\n'{yolo_viz_path}'"  
            else:  
                error_message += "未能显示YOLO分割结果（未知原因）。"
            print(f"[DEBUG] {error_message.replace('/n', ' ')}")

            no_result_label = tk.Label(self.roi_detail_frame,
                                       text=error_message,
                                       wraplength=max(200, self.roi_detail_frame.winfo_width() - 20),
                                       justify=tk.CENTER)
            no_result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        print("--- [DEBUG] display_roi_segmentation_detail finished ---")

    def _on_detection_window_close(self):
        print("[DEBUG Window] _on_detection_window_close called.")
        if self.fig_canvas and hasattr(self, 'click_event_cid') and self.click_event_cid is not None:
            try:
                self.fig_canvas.mpl_disconnect(self.click_event_cid)
                print(f"[DEBUG Window] Disconnected mpl_connect event with cid: {self.click_event_cid}")
                self.click_event_cid = None  
            except Exception as e:
                print(f"[ERROR Window] Error disconnecting canvas event: {e}")

        
        if hasattr(self, 'heatmap_ax') and self.heatmap_ax:
            
            self.heatmap_ax = None
            print("[DEBUG Window] self.heatmap_ax set to None.")

        if hasattr(self, 'fig_canvas') and self.fig_canvas:
            
            
            self.fig_canvas = None
            print("[DEBUG Window] self.fig_canvas set to None.")

        self.clickable_rois_info_for_event = []  

        if hasattr(self, 'detection_results_window') and self.detection_results_window.winfo_exists():
            self.detection_results_window.destroy()
            print("[DEBUG Window] detection_results_window destroyed.")


    def select_weight(self, model_type):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch files", "*.pth *.pt"), ("All files", "*.*")])
        if file_path:
            if model_type == "CNN":
                self.cnn_weight_path.set(file_path)
                self.cnn_weight_label.config(text=os.path.basename(file_path))
            elif model_type == "YOLO11":
                self.yolo_weight_path.set(file_path)
                self.yolo_weight_label.config(text=os.path.basename(file_path))

    def _show_detection_results(self, mask_data):
        if hasattr(self, 'detection_results_window') and self.detection_results_window.winfo_exists():
            self.detection_results_window.destroy()
            self.fig_canvas = None
            self.heatmap_ax = None
            self.clickable_rois_info_for_event = []

        self.detection_results_window = tk.Toplevel(self.master)
        self.detection_results_window.title("轮廓检测与模型应用结果")
        self.detection_results_window.geometry("1200x800")

        
        model_frame = tk.Frame(self.detection_results_window, padx=10, pady=10)
        model_frame.pack(fill=tk.X)

        tk.Label(model_frame, text="CNN权重:").pack(side=tk.LEFT)
        cnn_entry = tk.Entry(model_frame, textvariable=self.cnn_weight_path, width=40)
        cnn_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(model_frame, text="选择CNN权重", command=lambda: self.select_file_path(self.cnn_weight_path,
                                                                                         [("PyTorch Model",
                                                                                           "*.pt *.pth")])).pack(
            side=tk.LEFT, padx=5)  

        tk.Label(model_frame, text="YOLO权重:").pack(side=tk.LEFT, padx=(10, 0))
        yolo_entry = tk.Entry(model_frame, textvariable=self.yolo_weight_path, width=40)
        yolo_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(model_frame, text="选择YOLO权重",
                  command=lambda: self.select_file_path(self.yolo_weight_path, [("PyTorch Model", "*.pt")])).pack(
            side=tk.LEFT, padx=5)

        apply_btn = tk.Button(model_frame, text="应用模型", command=lambda: self.apply_models(mask_data))
        apply_btn.pack(side=tk.LEFT, padx=20, pady=5)

        
        main_content_pane = tk.PanedWindow(self.detection_results_window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED,
                                           bd=2)
        main_content_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.heatmap_display_frame = tk.Frame(main_content_pane, relief=tk.SUNKEN, borderwidth=1)
        main_content_pane.add(self.heatmap_display_frame, stretch="always", width=800)

        self.roi_detail_frame = tk.Frame(main_content_pane, relief=tk.SUNKEN, borderwidth=1)
        main_content_pane.add(self.roi_detail_frame, stretch="never", width=350)

        for widget in self.roi_detail_frame.winfo_children():
            widget.destroy()
        self.roi_detail_label = tk.Label(self.roi_detail_frame,
                                         text="点击左侧热图上橙色标记的区域\n(CNN分类为0)\n以在此处查看其YOLO分割详情。",
                                         wraplength=300, justify=tk.CENTER, padx=10, pady=10)
        self.roi_detail_label.pack(fill=tk.BOTH, expand=True)

        self.display_results(mask_data, show_segmentation=True, show_final_ml_results=False)  

        self.detection_results_window.protocol("WM_DELETE_WINDOW", self._on_detection_window_close)

    def _save_segmented_instances(self, mask_info, yolo_input_image):
        if "segmentation" not in mask_info or mask_info["segmentation"] is None:
            print(f"Mask {mask_info['id']}: 未找到分割数据，无法保存实例。")
            return
        
        mask_dir = os.path.dirname(mask_info["mask_path"])
        segmentation_masks_np = mask_info["segmentation"]
        num_instances = segmentation_masks_np.shape[0]
        saved_count = 0

        for i in range(num_instances):
            instance_mask_np = segmentation_masks_np[i]

            if instance_mask_np.max() <= 1.0 and instance_mask_np.min() >= 0.0:
                instance_mask_np = (instance_mask_np * 255).astype(np.uint8)
            else:
                instance_mask_np = instance_mask_np.astype(np.uint8)

            if instance_mask_np.shape[0] != yolo_input_image.shape[0] or \
                    instance_mask_np.shape[1] != yolo_input_image.shape[1]:
                instance_mask_np = cv2.resize(instance_mask_np,
                                              (yolo_input_image.shape[1], yolo_input_image.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)

            if len(yolo_input_image.shape) == 2:
                yolo_input_img_color = cv2.cvtColor(yolo_input_image, cv2.COLOR_GRAY2BGR)
            else:
                yolo_input_img_color = yolo_input_image.copy()

            _, binary_mask = cv2.threshold(instance_mask_np, 127, 255, cv2.THRESH_BINARY)
            segmented_instance = cv2.bitwise_and(yolo_input_img_color, yolo_input_img_color, mask=binary_mask)

            instance_filename = f"instance_{i:03d}.jpg"
            instance_save_path = os.path.join(mask_dir, instance_filename)
            try:
                cv2.imwrite(instance_save_path, segmented_instance)
                saved_count += 1
            except Exception as e:
                print(f"错误：保存分割实例 {instance_save_path} 失败: {e}")

        if saved_count > 0:
            print(f"Mask {mask_info['id']}: 成功保存 {saved_count} 个分割实例到目录 {mask_dir}")

    def show_annotated_heatmap(self, mask_data):
        if not mask_data or "original_heatmap" not in mask_data[0]:
            print("错误：mask_data 为空或未提供原始热图路径")
            return

        original_heatmap_path = os.path.join("./results", mask_data[0]["original_heatmap"])
        if not os.path.exists(original_heatmap_path):
            print(f"错误：原始热图文件 {original_heatmap_path} 不存在")
            return
        
        original_heatmap = cv2.imread(original_heatmap_path)
        if original_heatmap is None:
            print(f"错误：无法加载原始热图 {original_heatmap_path}")
            return
        original_heatmap = cv2.cvtColor(original_heatmap, cv2.COLOR_BGR2RGB)  

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(original_heatmap)

        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        for idx, mask in enumerate(mask_data):
            x, y, w, h = mask["original_bbox"]
            color = colors[idx % len(colors)]
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 10, f"Mask {mask['id']}", color=color)

        ax.set_title("原始热图与掩码位置")
        ax.axis('off')  

        result_window = tk.Toplevel(self.master)
        result_window.title("带掩码标注的热图")
        result_window.geometry("1000x800")

        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedChromatogramGUI(root)
    root.mainloop()