import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.ttk import PanedWindow
import cv2
import heapq
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import csv
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 0: PATH PROPERTIES GENERATOR (THE BRAIN)
# This class takes a clean geometric path (in meters) and figures out the ideal
# speed and curvature for each point. It's like a race engineer calculating the
# perfect line through a track, based only on physics.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PathPropertiesGenerator:
    """Calculates the ideal state (speed, curvature) for a given geometric path."""
    def __init__(self, planning_params):
        self.p = planning_params

    def _calculate_derivatives(self, x, y):
        dx, dy = np.gradient(x), np.gradient(y)
        self.ds = np.sqrt(dx**2 + dy**2)
        self.ds[self.ds < 1e-6] = 1e-6
        self.x_prime = np.gradient(x) / self.ds
        self.y_prime = np.gradient(y) / self.ds
        self.x_double_prime = np.gradient(self.x_prime) / self.ds
        self.y_double_prime = np.gradient(self.y_prime) / self.ds
        
    def _calculate_curvature_and_radius(self):
        numerator = self.x_prime * self.y_double_prime - self.y_prime * self.x_double_prime
        denominator = (self.x_prime**2 + self.y_prime**2)**1.5
        denominator[denominator < 1e-6] = 1e-6
        self.kappa = numerator / denominator
        self.radius = np.full_like(self.kappa, self.p['R_max'])
        mask = np.abs(self.kappa) > self.p['kappa_min']
        self.radius[mask] = 1.0 / np.abs(self.kappa[mask])

    def _calculate_effective_radius(self):
        weights_speed = np.ones(self.p['n_speed']) / self.p['n_speed']
        self.R_eff_speed = np.convolve(self.radius, weights_speed, mode='same')
        pad_speed = self.p['n_speed'] // 2
        self.R_eff_speed[:pad_speed] = self.radius[0]
        self.R_eff_speed[-pad_speed:] = self.radius[-1]
        
    def _calculate_curvature_speed_limit(self):
        a_max = self.p['safety_factor'] * self.p['mu'] * self.p['g']
        self.v_profile = np.sqrt(a_max * self.R_eff_speed)
        self.v_profile = np.minimum(self.v_profile, self.p['v_max'])
        self.v_profile[-1] = 0

    def generate_properties(self, path_points):
        if len(path_points) < 5: return None
        x, y = path_points[:, 0], path_points[:, 1]
        self._calculate_derivatives(x, y)
        self._calculate_curvature_and_radius()
        self._calculate_effective_radius()
        self._calculate_curvature_speed_limit()
        return (path_points, self.v_profile, self.kappa)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 1: THE ALCHEMY LAB (CORE IMAGE PROCESSING)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def skeletonize_image(image):
    """Whittles a shape down to its bare bones (a 1-pixel wide centerline)."""
    skeleton = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) != 0:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
    return skeleton

def preprocess_and_extract_path(image_path, lower_bound, upper_bound):
    """The master plan for path extraction from an image."""
    img = cv2.imread(image_path)
    if img is None: return None, None, None, 1.0
    original_img = cv2.imread(image_path)

    scale_factor = 1.0
    if img.shape[1] > 1000:
        scale_percent = 50; scale_factor = scale_percent / 100.0
        width = int(img.shape[1] * scale_factor); height = int(img.shape[0] * scale_factor)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return original_img, None, mask, scale_factor

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    erosion_kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(clean_mask, erosion_kernel, iterations=1)
    
    skeleton = skeletonize_image(eroded_mask)

    rows, cols = np.where(skeleton > 0)
    if len(rows) < 2: return original_img, None, skeleton, scale_factor

    all_points_scaled = np.column_stack((cols, rows))
    return original_img, all_points_scaled, clean_mask, scale_factor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2: POP-UP WINDOWS OF WONDER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MaskViewer(tk.Toplevel):
    def __init__(self, parent, mask_image):
        super().__init__(parent); self.title("Full Detected Mask"); self.geometry("600x600")
        self.mask_image = mask_image; self.mask_label = tk.Label(self); self.mask_label.pack(fill=tk.BOTH, expand=True)
        self.bind("<Configure>", self.on_resize)
    def on_resize(self, event=None):
        container_w, container_h = self.winfo_width(), self.winfo_height()
        if container_w < 2 or container_h < 2: return
        img_h, img_w = self.mask_image.shape; scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(self.mask_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)

class ColorTunerWindow(tk.Toplevel):
    def __init__(self, parent, image_path, current_lower, current_upper, on_apply_callback):
        super().__init__(parent); self.title("Color Tuner"); self.geometry("800x700")
        self.on_apply_callback = on_apply_callback
        img = cv2.imread(image_path); img_for_hsv = img.copy()
        if img.shape[1] > 1000:
            scale_percent = 50; width = int(img.shape[1] * scale_percent / 100); height = int(img.shape[0] * scale_percent / 100)
            img_for_hsv = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        self.hsv = cv2.cvtColor(img_for_hsv, cv2.COLOR_BGR2HSV)
        controls_container = tk.Frame(self); controls_container.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
        self.mask_label = tk.Label(self, bg="black"); self.mask_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        sliders_frame = tk.Frame(controls_container); sliders_frame.pack(fill=tk.X)
        self.h_min = tk.Scale(sliders_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="H Min", command=self.update_mask); self.h_min.set(current_lower[0]); self.h_min.pack(fill=tk.X)
        self.s_min = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="S Min", command=self.update_mask); self.s_min.set(current_lower[1]); self.s_min.pack(fill=tk.X)
        self.v_min = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="V Min", command=self.update_mask); self.v_min.set(current_lower[2]); self.v_min.pack(fill=tk.X)
        self.h_max = tk.Scale(sliders_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="H Max", command=self.update_mask); self.h_max.set(current_upper[0]); self.h_max.pack(fill=tk.X)
        self.s_max = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="S Max", command=self.update_mask); self.s_max.set(current_upper[1]); self.s_max.pack(fill=tk.X)
        self.v_max = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="V Max", command=self.update_mask); self.v_max.set(current_upper[2]); self.v_max.pack(fill=tk.X)
        btn_frame = tk.Frame(controls_container, pady=5); btn_frame.pack()
        tk.Button(btn_frame, text="Apply", command=self.apply_and_close).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        self.after(100, self.update_mask)
    def update_mask(self, val=None):
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]); upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        mask = cv2.inRange(self.hsv, lower, upper)
        container_w, container_h = self.mask_label.winfo_width(), self.mask_label.winfo_height()
        if container_w < 2 or container_h < 2: self.after(100, self.update_mask); return
        img_h, img_w = mask.shape; scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)
    def apply_and_close(self):
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]); upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        self.on_apply_callback(lower, upper); self.destroy()

class InteractiveSlopeWindow(tk.Toplevel):
    def __init__(self, parent, original_cv_image, path_points, tck, u_params):
        super().__init__(parent); self.title("Interactive Slope Analysis"); self.geometry("1300x700")
        self.original_cv_image = original_cv_image; self.path_points, self.tck, self.u_params = path_points, tck, u_params
        x_range = np.ptp(self.path_points[:, 0]); y_range = np.ptp(self.path_points[:, 1]); self.is_math_plot_rotated = y_range > x_range
        plots_frame = tk.Frame(self); plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        controls_frame = tk.Frame(self, pady=10); controls_frame.pack(fill=tk.X, expand=False, padx=10)
        plot_paned_window = PanedWindow(plots_frame, orient=tk.HORIZONTAL); plot_paned_window.pack(fill=tk.BOTH, expand=True)
        img_plot_frame = tk.Frame(plot_paned_window); plot_paned_window.add(img_plot_frame, weight=1)
        self.fig_img = Figure(); self.ax_img = self.fig_img.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=img_plot_frame); self.canvas_img.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        math_plot_frame = tk.Frame(plot_paned_window); plot_paned_window.add(math_plot_frame, weight=1)
        self.fig_math = Figure(); self.ax_math = self.fig_math.add_subplot(111)
        self.canvas_math = FigureCanvasTkAgg(self.fig_math, master=math_plot_frame); self.canvas_math.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        data_frame = tk.Frame(controls_frame); data_frame.pack(side=tk.LEFT, padx=(0, 20))
        self.slope_var = tk.StringVar(value="Slope: -"); self.int_slope_var = tk.StringVar(value="Integer Slope: -"); self.angle_var = tk.StringVar(value="Angle: -")
        tk.Label(data_frame, textvariable=self.slope_var, font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(data_frame, textvariable=self.int_slope_var, font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(data_frame, textvariable=self.angle_var, font=("Helvetica", 11)).pack(anchor="w")
        slider_frame = tk.Frame(controls_frame); slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.slider = tk.Scale(slider_frame, from_=0, to=len(self.u_params) - 1, orient=tk.HORIZONTAL, command=self._update_plots, label="Position Along Path"); self.slider.pack(fill=tk.X, expand=True)
        self._draw_initial_plots(); self.after(100, lambda: self.slider.set(len(self.path_points) // 2))
    def _draw_initial_plots(self):
        img_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(img_rgb, aspect='equal'); self.ax_img.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2.5, alpha=0.8)
        self.point_on_img, = self.ax_img.plot([], [], 'o', color='magenta', markersize=12, markeredgecolor='black', zorder=10)
        self.tangent_on_img, = self.ax_img.plot([], [], color='yellow', linestyle='--', linewidth=2.5, zorder=9)
        self.ax_img.set_title("Image Overlay View"); self.ax_img.axis('off'); self.fig_img.tight_layout()
        if self.is_math_plot_rotated:
            self.ax_math.plot(self.path_points[:, 1], self.path_points[:, 0], color='red', linewidth=2); self.ax_math.set_xlabel("Primary Axis (Y pixels)"); self.ax_math.set_ylabel("Secondary Axis (X pixels)")
        else:
            self.ax_math.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2); self.ax_math.set_xlabel("Primary Axis (X pixels)"); self.ax_math.set_ylabel("Secondary Axis (Y pixels)")
        self.point_on_math, = self.ax_math.plot([], [], 'o', color='magenta', markersize=10, markeredgecolor='black', zorder=10)
        self.tangent_on_math, = self.ax_math.plot([], [], color='blue', linestyle='--', linewidth=2, zorder=9)
        self.ax_math.set_aspect('equal', adjustable='box'); self.ax_math.invert_yaxis(); self.ax_math.grid(True, linestyle='--', alpha=0.6)
        self.ax_math.set_title("Mathematical View"); self.fig_math.tight_layout(); self.canvas_img.draw(); self.canvas_math.draw()
    def _update_plots(self, val):
        index = int(val); u_val = self.u_params[index]; current_point = splev(u_val, self.tck, der=0)
        dx_dt, dy_dt = splev(u_val, self.tck, der=1); cx, cy = current_point[0], current_point[1]
        if self.is_math_plot_rotated:
            if dy_dt < 0: dx_dt, dy_dt = -dx_dt, -dy_dt
            primary_deriv, secondary_deriv = dy_dt, dx_dt; slope_label_text = "Slope (dX/dY)"
        else:
            if dx_dt < 0: dx_dt, dy_dt = -dx_dt, -dy_dt
            primary_deriv, secondary_deriv = dx_dt, dy_dt; slope_label_text = "Slope (dY/dX)"
        angle_deg = np.degrees(np.arctan2(secondary_deriv, primary_deriv))
        if abs(primary_deriv) < 1e-6:
            self.slope_var.set(f"{slope_label_text}: Infinity"); self.int_slope_var.set("Integer Slope: N/A")
        else:
            slope_val = secondary_deriv / primary_deriv; self.slope_var.set(f"{slope_label_text}: {slope_val:.2f}"); self.int_slope_var.set(f"Integer Slope: {int(round(slope_val))}")
        self.angle_var.set(f"Angle: {angle_deg:.1f}°")
        magnitude = np.hypot(dx_dt, dy_dt); line_length = 50
        if magnitude > 1e-6:
            ux, uy = dx_dt / magnitude, dy_dt / magnitude; x_coords = [cx - ux * line_length, cx + ux * line_length]; y_coords = [cy - uy * line_length, cy + uy * line_length]
        else: x_coords, y_coords = [], []
        self.point_on_img.set_data([cx], [cy]); self.tangent_on_img.set_data(x_coords, y_coords)
        if self.is_math_plot_rotated: self.point_on_math.set_data([cy], [cx]); self.tangent_on_math.set_data(y_coords, x_coords)
        else: self.point_on_math.set_data([cx], [cy]); self.tangent_on_math.set_data(x_coords, y_coords)
        self.canvas_img.draw_idle(); self.canvas_math.draw_idle()

class InteractiveSegmentTuner(tk.Toplevel):
    def __init__(self, parent, path_data, on_generate_callback):
        super().__init__(parent)
        self.title("Interactive Segment Tuner")
        self.geometry("1000x800")

        self.path_data = path_data
        self.on_generate_callback = on_generate_callback
        self.segments = []
        
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.LabelFrame(main_frame, text="Tuning Controls", padx=10, pady=10)
        controls_frame.pack(fill=tk.X, pady=(10, 0))

        self.fig = Figure(); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.min_len_var = tk.DoubleVar(value=0.05)
        
        thresh_frame = tk.Frame(controls_frame); thresh_frame.pack(fill=tk.X, expand=True, pady=2)
        tk.Label(thresh_frame, text="Straightness Threshold (1/radius):", width=25, anchor='w').pack(side=tk.LEFT)
        self.thresh_slider = tk.Scale(thresh_frame, from_=0.0, to=5.0, resolution=0.05,
                                      orient=tk.HORIZONTAL, variable=self.threshold_var, command=self.update_segmentation)
        self.thresh_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(thresh_frame, textvariable=self.threshold_var, width=5).pack(side=tk.LEFT, padx=(5,0))

        min_len_frame = tk.Frame(controls_frame); min_len_frame.pack(fill=tk.X, expand=True, pady=2)
        tk.Label(min_len_frame, text="Minimum Segment Length (meters):", width=25, anchor='w').pack(side=tk.LEFT)
        self.min_len_slider = tk.Scale(min_len_frame, from_=0.0, to=0.5, resolution=0.01,
                                     orient=tk.HORIZONTAL, variable=self.min_len_var, command=self.update_segmentation)
        self.min_len_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(min_len_frame, textvariable=self.min_len_var, width=5).pack(side=tk.LEFT, padx=(5,0))

        tk.Button(controls_frame, text="Generate PATH.TXT", font=("Helvetica", 12, "bold"),
                  bg="#ccffcc", command=self._finalize_and_export).pack(fill=tk.X, ipady=5, pady=(10,0))
        
        self.after(100, self.update_segmentation)

    def update_segmentation(self, val=None):
        self.segments = self._run_segmentation(0, len(self.path_data['kappa']))
        self._draw_segments()

    def _run_segmentation(self, start_idx, end_idx):
        if end_idx <= start_idx: return []
        
        straight_threshold = self.threshold_var.get()
        min_segment_len = self.min_len_var.get()
        
        segments = []
        current_state = "IDLE"
        segment_start_index_rel = 0

        for i in range(end_idx - start_idx):
            current_abs_idx = start_idx + i
            point_is_turn = abs(self.path_data['kappa'][current_abs_idx]) > straight_threshold
            
            transitioned, end_state = False, None
            if current_state == "IDLE":
                current_state = "IN_TURN" if point_is_turn else "IN_STRAIGHT"
                segment_start_index_rel = i
            elif current_state == "IN_STRAIGHT" and point_is_turn:
                transitioned, end_state = True, "IN_STRAIGHT"
            elif current_state == "IN_TURN" and not point_is_turn:
                transitioned, end_state = True, "IN_TURN"

            if transitioned:
                abs_start, abs_end = start_idx + segment_start_index_rel, start_idx + i
                seg_dist = np.sum(self.path_data['step_distances'][abs_start : abs_end-1])
                if seg_dist >= min_segment_len:
                    segments.append(self._create_segment_dict(abs_start, abs_end, end_state))
                current_state = "IN_TURN" if point_is_turn else "IN_STRAIGHT"
                segment_start_index_rel = i

        abs_start, abs_end = start_idx + segment_start_index_rel, end_idx
        last_seg_dist = np.sum(self.path_data['step_distances'][abs_start:abs_end-1])
        if last_seg_dist >= min_segment_len:
            segments.append(self._create_segment_dict(abs_start, abs_end, current_state))
        return segments

    def _create_segment_dict(self, start_idx, end_idx, seg_type):
        seg_dist = np.sum(self.path_data['step_distances'][start_idx : end_idx-1])
        segment_points_pixels = self.path_data['points_pixels'][start_idx:end_idx]
        
        if seg_type == "IN_STRAIGHT":
            speed, curvature, viz_type = self.path_data['v_max'], 0.0, 'STRAIGHT'
        else: # IN_TURN
            speed = np.mean(self.path_data['v_profile'][start_idx:end_idx])
            curvature = np.mean(self.path_data['kappa'][start_idx:end_idx])
            viz_type = 'ARC'
        
        return {'type': viz_type, 'points': segment_points_pixels, 'dist': seg_dist,
                'speed': speed, 'curvature': curvature, 'start_idx': start_idx, 'end_idx': end_idx}

    def _draw_segments(self):
        self.ax.clear()
        self.ax.set_title(f"Segmentation ({len(self.segments)} segments)")
        self.ax.set_aspect('equal', 'box'); self.ax.invert_yaxis()
        self.ax.grid(True, linestyle='--', alpha=0.6)

        colors = {'STRAIGHT': ['#00FFFF', '#008B8B'], 'ARC': ['#FF00FF', '#8B008B']}
        color_idx = {'STRAIGHT': 0, 'ARC': 0}
        
        for seg in self.segments:
            points = seg['points']
            seg_type = seg['type']
            if len(points) > 0:
                color = colors[seg_type][color_idx[seg_type]]
                self.ax.plot(points[:, 0], points[:, 1], color=color, linewidth=4)
                color_idx[seg_type] = 1 - color_idx[seg_type]
        
        self.fig.tight_layout()
        self.canvas.draw()

    def _finalize_and_export(self):
        if not self.segments:
            messagebox.showwarning("Warning", "No segments generated. Try adjusting sliders.", parent=self)
            return
        
        self.on_generate_callback(self.segments)
        self.destroy()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 3: THE MAIN APPLICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PathAnalyzerApp:
    def __init__(self, root):
        self.root = root; self.root.title("Path Analyzer 9000"); self.root.geometry("1200x800")
        self.image_path = None; self.original_cv_img = None; self.final_mask = None; self.raw_skeleton_points = None
        self.all_ordered_points = None; self.active_ordered_points = None; self.displayed_nodes = None
        self.smooth_path_points = None; self.start_node = None; self.tck = None; self.u_params = None
        self.hsv_lower = np.array([0, 70, 50]); self.hsv_upper = np.array([179, 255, 255])
        self.selection_mode = "none"; self.manual_start_point = None; self.manual_end_point = None
        self.click_handler_id = None; self.motion_handler_id = None; self.hovered_node_marker = None
        self.currently_hovered_node_index = None
        self.pixels_per_meter = None; self.calibration_mode = "none"; self.calib_point1 = None; self.calib_line = None
        self._build_gui()

    def _build_gui(self):
        self.main_paned_window = PanedWindow(self.root, orient=tk.HORIZONTAL); self.main_paned_window.pack(fill=tk.BOTH, expand=True)
        self.left_pane = tk.Frame(self.main_paned_window); self.main_paned_window.add(self.left_pane, weight=3)
        self.right_paned_window = PanedWindow(self.main_paned_window, orient=tk.VERTICAL); self.main_paned_window.add(self.right_paned_window, weight=1)
        self.fig = Figure(); self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_pane); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        controls_frame = tk.Frame(self.right_paned_window); math_plot_frame = tk.Frame(self.right_paned_window)
        self.right_paned_window.add(controls_frame, weight=0); self.right_paned_window.add(math_plot_frame, weight=1)
        
        workflow_frame = tk.LabelFrame(controls_frame, text="Workflow Steps", padx=10, pady=10); workflow_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        self.btn_load = tk.Button(workflow_frame, text="1. Load Image", command=self.load_image); self.btn_load.pack(fill=tk.X, pady=2)
        self.btn_tuner = tk.Button(workflow_frame, text="2. Tune Color Range...", command=self.open_color_tuner, state=tk.DISABLED); self.btn_tuner.pack(fill=tk.X, pady=2)
        self.btn_analyze = tk.Button(workflow_frame, text="3. Analyze & Select Endpoints", command=self.analyze_path, state=tk.DISABLED); self.btn_analyze.pack(fill=tk.X, pady=(2, 0))
        
        tuning_frame = tk.LabelFrame(controls_frame, text="Path Visualization Tuning", padx=10, pady=5); tuning_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(tuning_frame, text="Path Endpoint Control (Fine-tuning):").pack()
        self.path_length_var = tk.IntVar(value=100)
        self.path_length_slider = tk.Scale(tuning_frame, from_=2, to=100, orient=tk.HORIZONTAL, variable=self.path_length_var, command=self._update_path_length, state=tk.DISABLED, showvalue=0); self.path_length_slider.pack(fill=tk.X)
        tk.Label(tuning_frame, text="Number of Nodes to Display:").pack()
        self.nodes_var = tk.IntVar(value=150)
        self.nodes_slider = tk.Scale(tuning_frame, from_=2, to=1000, orient=tk.HORIZONTAL, variable=self.nodes_var, command=self.update_node_display_during_selection, showvalue=0); self.nodes_slider.pack(fill=tk.X)
        
        export_frame = tk.LabelFrame(controls_frame, text="Export for Vehicle Control", padx=10, pady=10); export_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_export = tk.Button(export_frame, text="4. Tune and Export Path...", command=self.export_path_data, state=tk.DISABLED, bg="#ccffcc")
        self.btn_export.pack(fill=tk.X, ipady=5)
        
        tools_frame = tk.LabelFrame(controls_frame, text="Diagnostic Tools", padx=10, pady=5); tools_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_calibrate = tk.Button(tools_frame, text="Calibrate Pixel/Meter Scale...", command=self.start_scale_calibration, state=tk.DISABLED); self.btn_calibrate.pack(fill=tk.X, pady=2)
        self.btn_show_mask = tk.Button(tools_frame, text="Show Full Mask", command=self.show_full_mask, state=tk.DISABLED); self.btn_show_mask.pack(fill=tk.X, pady=2)
        self.btn_slope = tk.Button(tools_frame, text="Interactive Slope Analysis", command=self.open_slope_visualizer, state=tk.DISABLED); self.btn_slope.pack(fill=tk.X, pady=2)
        
        self.status_label = tk.Label(controls_frame, text="Please load an image to begin.", wraplength=350, justify=tk.LEFT); self.status_label.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        tk.Label(math_plot_frame, text="Mathematical 2D Plot").pack()
        self.math_fig = Figure(); self.math_ax = self.math_fig.add_subplot(111)
        self.math_canvas = FigureCanvasTkAgg(self.math_fig, master=math_plot_frame); self.math_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

    def _disconnect_event_handlers(self):
        if self.click_handler_id: self.canvas.mpl_disconnect(self.click_handler_id); self.click_handler_id = None
        if self.motion_handler_id: self.canvas.mpl_disconnect(self.motion_handler_id); self.motion_handler_id = None
        self._remove_hover_marker()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not path: return
        self.image_path = path; self.original_cv_img = cv2.imread(self.image_path)
        if self.original_cv_img is None: messagebox.showerror("Error", "Failed to load image."); return
        self.pixels_per_meter = None; self.raw_skeleton_points = None; self.all_ordered_points = None; self.active_ordered_points = None
        self.display_image(self.original_cv_img, "Original Image Loaded")
        self.update_math_plot()
        self.btn_tuner.config(state=tk.NORMAL); self.btn_calibrate.config(state=tk.NORMAL); self.btn_analyze.config(state=tk.DISABLED)
        self.btn_show_mask.config(state=tk.DISABLED); self.btn_slope.config(state=tk.DISABLED); self.path_length_slider.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)
        self._disconnect_event_handlers()
        self.status_label.config(text="Image loaded. Calibrate scale or tune colors.")

    def analyze_path(self):
        self.status_label.config(text="Analyzing image to find path..."); self.root.update_idletasks()
        self.btn_analyze.config(state=tk.DISABLED)
        self.original_cv_img, raw_points, self.final_mask, scale_factor = preprocess_and_extract_path(self.image_path, self.hsv_lower, self.hsv_upper)
        if raw_points is None or len(raw_points) < 20:
            messagebox.showerror("Error", "Could not find a significant path. Try tuning the colors."); self.status_label.config(text="Analysis failed.")
            self.btn_analyze.config(state=tk.NORMAL); return
        self.raw_skeleton_points = (raw_points / scale_factor).astype(int)
        self.smooth_path_points = None
        self.update_node_display_during_selection()
        self.selection_mode = "start"; self.manual_start_point, self.manual_end_point = None, None
        self.click_handler_id = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.motion_handler_id = self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.status_label.config(text="Path detected! Please click on the START node.")

    def start_scale_calibration(self):
        self._disconnect_event_handlers(); self.selection_mode = "none"
        self.calibration_mode = "point1"; self.calib_point1 = None
        if self.calib_line: self.calib_line.remove(); self.calib_line = None; self.canvas.draw()
        self.status_label.config(text="Calibration Mode: Click the START point of your reference object.")
        self.click_handler_id = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _on_canvas_click(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        if self.calibration_mode == "point1":
            self.calib_point1 = (event.xdata, event.ydata)
            self.status_label.config(text="Got it. Now click the END point of your reference object.")
            self.calibration_mode = "point2"; return
        elif self.calibration_mode == "point2":
            calib_point2 = (event.xdata, event.ydata)
            if self.calib_line: self.calib_line.remove()
            self.calib_line, = self.ax.plot([self.calib_point1[0], calib_point2[0]], [self.calib_point1[1], calib_point2[1]], 'y-', linewidth=3, marker='o')
            self.canvas.draw()
            pixel_dist = np.hypot(calib_point2[0] - self.calib_point1[0], calib_point2[1] - self.calib_point1[1])
            real_dist_m = simpledialog.askfloat("Input", "What is the real-world length of the drawn line (in meters)?", parent=self.root, minvalue=0.001)
            if real_dist_m and real_dist_m > 0:
                self.pixels_per_meter = pixel_dist / real_dist_m
                self.status_label.config(text=f"Scale calibrated: {self.pixels_per_meter:.2f} pixels per meter.")
            else: self.status_label.config(text="Scale calibration cancelled.")
            self.calibration_mode = "none"; self._disconnect_event_handlers(); return
        if self.selection_mode == "none" or self.currently_hovered_node_index is None: return
        selected_node = self.displayed_nodes[self.currently_hovered_node_index]
        if self.selection_mode == "start":
            self.manual_start_point = selected_node; self.selection_mode = "end"; self.status_label.config(text="Start node locked! Now, click on the END node.")
        elif self.selection_mode == "end":
            self.manual_end_point = selected_node; self.selection_mode = "none"; self._disconnect_event_handlers()
            self.status_label.config(text="Endpoints selected. Crunching numbers..."); self.root.after(50, self._process_manual_endpoints)
    
    def _on_canvas_motion(self, event):
        if event.inaxes != self.ax or self.displayed_nodes is None or len(self.displayed_nodes) == 0: self._remove_hover_marker(); return
        mouse_point = np.array([event.xdata, event.ydata]); distances = np.linalg.norm(self.displayed_nodes - mouse_point, axis=1)
        closest_node_index = np.argmin(distances); min_dist = distances[closest_node_index]
        if min_dist < 15:
            if self.currently_hovered_node_index != closest_node_index:
                self._remove_hover_marker(); node_coords = self.displayed_nodes[closest_node_index]
                self.hovered_node_marker, = self.ax.plot(node_coords[0], node_coords[1], 'o', markersize=18, color='yellow', alpha=0.7, zorder=10)
                self.currently_hovered_node_index = closest_node_index; self.canvas.draw_idle()
        else: self._remove_hover_marker()
            
    def _remove_hover_marker(self):
        if self.hovered_node_marker: self.hovered_node_marker.remove(); self.hovered_node_marker = None; self.currently_hovered_node_index = None; self.canvas.draw_idle()
            
    def _process_manual_endpoints(self):
        self.status_label.config(text="Ordering path points..."); self.root.update_idletasks()
        self.all_ordered_points = self.find_shortest_path(self.raw_skeleton_points, self.manual_start_point, self.manual_end_point)

        if self.all_ordered_points is None:
            messagebox.showerror("Pathfinding Error", "Could not find a valid path between the selected start and end points. Please try again.")
            self.status_label.config(text="Pathfinding failed. Please select new endpoints.")
            self.analyze_path()
            return
            
        self.btn_show_mask.config(state=tk.NORMAL)
        num_total_points = len(self.all_ordered_points)
        self.path_length_slider.config(to=num_total_points, state=tk.NORMAL); self.path_length_var.set(num_total_points)
        self._update_path_length()
        self.status_label.config(text="Path analysis complete! Adjust sliders or export."); self.btn_analyze.config(state=tk.NORMAL)

    def find_shortest_path(self, points, start_point, end_point):
        if len(points) < 2: return points
        start_idx = np.argmin(np.linalg.norm(points - start_point, axis=1))
        end_idx = np.argmin(np.linalg.norm(points - end_point, axis=1))
        num_points = len(points)
        dist = np.full(num_points, np.inf)
        prev = np.full(num_points, -1, dtype=int)
        dist[start_idx] = 0
        pq = [(0, start_idx)]
        visited_indices = set()
        while pq:
            d, u_idx = heapq.heappop(pq)
            if u_idx in visited_indices: continue
            visited_indices.add(u_idx)
            if u_idx == end_idx: break
            u_coords = points[u_idx]
            diffs = points - u_coords
            dists_sq = np.sum(diffs**2, axis=1)
            neighbor_indices = np.where((dists_sq > 0) & (dists_sq < 5**2))[0] 
            for v_idx in neighbor_indices:
                if v_idx not in visited_indices:
                    edge_weight = np.sqrt(dists_sq[v_idx])
                    if dist[u_idx] + edge_weight < dist[v_idx]:
                        dist[v_idx] = dist[u_idx] + edge_weight
                        prev[v_idx] = u_idx
                        heapq.heappush(pq, (dist[v_idx], v_idx))
        path = []
        curr_idx = end_idx
        if prev[curr_idx] != -1 or curr_idx == start_idx:
            while curr_idx != -1:
                path.append(points[curr_idx])
                curr_idx = prev[curr_idx]
            return np.array(path[::-1])
        return None
        
    def _update_path_length(self, val=None):
        if self.all_ordered_points is None: return
        length = self.path_length_var.get(); self.active_ordered_points = self.all_ordered_points[:length]
        if len(self.active_ordered_points) < 4:
            self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)
            self.update_node_display_during_selection(); return
        try:
            x, y = self.active_ordered_points[:, 0], self.active_ordered_points[:, 1]
            smoothing_factor = len(self.active_ordered_points) * 0.1
            self.tck, u = splprep([x, y], s=smoothing_factor, k=3)
            if self.tck:
                num_smooth_points = len(self.active_ordered_points) * 2; self.u_params = np.linspace(u.min(), u.max(), num_smooth_points)
                x_final, y_final = splev(self.u_params, self.tck); self.smooth_path_points = np.column_stack((x_final, y_final))
                self.btn_slope.config(state=tk.NORMAL); self.btn_export.config(state=tk.NORMAL)
            else: raise ValueError("Spline creation failed.")
        except Exception:
            self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)
        self.update_node_display_during_selection()

    def update_node_display_during_selection(self, val=None):
        points_to_display_from = self.active_ordered_points if self.active_ordered_points is not None else self.raw_skeleton_points
        if points_to_display_from is None or len(points_to_display_from) == 0:
            self.displayed_nodes = None; self.start_node = None
        else:
            num_nodes = max(2, min(self.nodes_var.get(), len(points_to_display_from)))
            indices = np.linspace(0, len(points_to_display_from) - 1, num_nodes, dtype=int)
            self.displayed_nodes = points_to_display_from[indices]
            self.start_node = self.active_ordered_points[0] if self.active_ordered_points is not None else None
        self.display_graph(); self.update_math_plot()

    def display_graph(self):
        self.ax.clear(); img_rgb = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal')
        if self.smooth_path_points is not None: self.ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2.5, label='Fitted Smooth Path', zorder=4)
        if self.displayed_nodes is not None: self.ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=35, zorder=5, edgecolors='black', label='Nodes')
        if self.start_node is not None: self.ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=150, zorder=6, edgecolors='black', label='Start')
        self.ax.legend(); self.ax.set_title("Path Node Overlay"); self.ax.axis('off'); self.canvas.draw()
        
    def update_math_plot(self):
        self.math_ax.clear(); self.math_ax.set_title("Mathematical 2D Plot"); self.math_ax.grid(True, linestyle='--', alpha=0.6)
        if self.smooth_path_points is None: self.math_canvas.draw(); return
        x_range = np.ptp(self.smooth_path_points[:, 0]); y_range = np.ptp(self.smooth_path_points[:, 1]); is_rotated = y_range > x_range
        if is_rotated:
            self.math_ax.plot(self.smooth_path_points[:, 1], self.smooth_path_points[:, 0], 'r-'); self.math_ax.set_xlabel("Y pixels"); self.math_ax.set_ylabel("X pixels")
        else:
            self.math_ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], 'r-'); self.math_ax.set_xlabel("X pixels"); self.math_ax.set_ylabel("Y pixels")
        self.math_ax.set_aspect('equal', adjustable='box'); self.math_ax.invert_yaxis(); self.math_fig.tight_layout(); self.math_canvas.draw()
    
    def on_tuner_apply(self, new_lower, new_upper):
        self.hsv_lower, self.hsv_upper = new_lower, new_upper; self.status_label.config(text="New color range applied. You may now Analyze the path.")
        self.btn_analyze.config(state=tk.NORMAL)
    
    def open_color_tuner(self):
        if self.image_path: ColorTunerWindow(self.root, self.image_path, self.hsv_lower, self.hsv_upper, self.on_tuner_apply)
    
    def open_slope_visualizer(self):
        if self.smooth_path_points is not None and self.tck is not None:
            InteractiveSlopeWindow(self.root, self.original_cv_img, self.smooth_path_points, self.tck, self.u_params)
        
    def display_image(self, cv_img, title=""):
        self.ax.clear(); img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal'); self.ax.set_title(title); self.ax.axis('off'); self.canvas.draw()
    
    def show_full_mask(self):
        if self.final_mask is not None: MaskViewer(self.root, self.final_mask)
        else: messagebox.showinfo("Info", "No mask has been generated yet.")

    def _on_tuner_generate(self, segments_to_export):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialfile="PATH.TXT", title="Save Arduino Path Command File"
        )
        if not file_path:
            self.status_label.config(text="Export cancelled.")
            return
        try:
            with open(file_path, 'w') as f:
                f.write("# Arduino Path Commands (Adaptive Segments)\n")
                f.write("# Format: DISTANCE,SPEED,CURVATURE\n")
                for seg in segments_to_export:
                    f.write(f"{seg['dist']:.4f},{seg['speed']:.4f},{seg['curvature']:.4f}\n")
            messagebox.showinfo("Success", f"Path successfully exported into {len(segments_to_export)} adaptive segments.\n\nFile saved to: {os.path.basename(file_path)}")
            self.status_label.config(text="Adaptive export complete!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write path file: {e}", parent=self.root)
            self.status_label.config(text="An error occurred during file writing.")

    def export_path_data(self):
        if self.tck is None or self.u_params is None:
            messagebox.showerror("Error", "No valid smooth path to export.")
            return
        if self.pixels_per_meter is None:
            messagebox.showerror("Error", "Scale not calibrated! Use 'Calibrate Pixel/Meter Scale'.")
            return
        self.status_label.config(text="Calculating high-resolution path properties..."); self.root.update_idletasks()
        try:
            planning_params = {
                'g': 9.81, 'mu': 0.8, 'safety_factor': 0.7, 'v_max': 2.5,
                'n_speed': 40, 'kappa_min': 1e-6, 'R_max': 10000.0,
            }
            num_points = 1000 
            u_vals_export = np.linspace(self.u_params.min(), self.u_params.max(), num_points)
            points_pixels = np.column_stack(splev(u_vals_export, self.tck))
            points_meters = points_pixels / self.pixels_per_meter

            prop_generator = PathPropertiesGenerator(planning_params)
            result = prop_generator.generate_properties(points_meters)
            
            if result is None: raise ValueError("Path is too short to generate properties.")

            _, v_profile, kappa = result
            diffs = np.diff(points_meters, axis=0)
            step_distances = np.sqrt(np.sum(diffs**2, axis=1))

            path_data_for_tuner = {
                'points_pixels': points_pixels, 'points_meters': points_meters,
                'v_profile': v_profile, 'kappa': kappa, 'step_distances': step_distances,
                'v_max': planning_params['v_max']
            }
            
            self.status_label.config(text="Ready for tuning. Opening tuner window...")
            InteractiveSegmentTuner(self.root, path_data_for_tuner, self._on_tuner_generate)

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to prepare data for tuner: {e}")
            self.status_label.config(text="An error occurred during export preparation.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 4: THE IGNITION SWITCH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    root = tk.Tk()
    app = PathAnalyzerApp(root)
    root.mainloop()