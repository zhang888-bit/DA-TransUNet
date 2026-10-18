import argparse
import logging
import shutil
import traceback
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import sys
import cv2
import time
import os
import copy
import numpy as np
import torch
from PIL import Image # 使用PIL读取图像，更好地处理不同格式


from app_logging import configure_logging, log_event
from clinical_validation import DEFAULT_REPORT_HTML, load_latest_report
from model_registry import get_model, list_models
from monitoring import get_system_health
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from report_export import export_analysis_report
from project_paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_VIT_NAME,
    LOG_ROOT,
    UI_ASSET_ROOT,
    UI_RESULT_ROOT,
    UI_TMP_ROOT,
    resolve_path,
    resolve_torch_device,
)

# --- Synapse 数据集特定配置 ---
NUM_CLASSES = 9 # 8个器官 + 1个背景
CLASS_NAMES = [
    "背景", "脾脏", "右肾", "左肾", "胆囊",
    "肝脏", "胃", "主动脉", "胰腺" # 注意：根据实际数据集调整
]
# 定义9个类别的颜色映射 (RGB格式) - 与数据处理脚本保持一致
COLORS = [
    (0, 0, 0),        # 0: 背景 - 黑色
    (128, 174, 128),  # 1: Spleen - 绿色
    (0, 0, 142),      # 2: Right Kidney - 深蓝
    (70, 130, 180),   # 3: Left Kidney - 钢蓝
    (255, 111, 111),  # 4: Gallbladder - 粉红
    (244, 35, 232),   # 6: Liver - 品红 (原颜色列表跳过了5, 假设映射到索引6)
    (220, 20, 60),    # 7: Stomach - 猩红
    (255, 0, 0),      # 8: Aorta - 红色
    (0, 0, 230),      # 9: Pancreas - 蓝色 (注意: 原始Synapse可能只有8个前景类, 索引可能映射到8)
    # 确保颜色列表长度至少为 NUM_CLASSES
]
IMG_SIZE = 224 # TransUNet模型期望的输入尺寸
MODEL_CONFIG_NAME = DEFAULT_VIT_NAME # 选择使用的ViT模型配置，需与训练时一致
# --------------------------

# 系统信息和界面文本 (更新为腹部器官相关)
SYSTEM_INFO = (f"腹部多器官信息 ({len(CLASS_NAMES)-1} 类):\n"
               f"类别: {', '.join(CLASS_NAMES[1:])}\n\n"
               "分割结果有助于可视化器官形态和位置。\n"
               "潜在应用:\n"
               "1. 辅助手术规划。\n"
               "2. 监测疾病进展或治疗反应。\n"
               "3. 医学教育与研究。\n\n"
               "免责声明: 本工具仅提供分割可视化功能，并非诊断工具。"
               "任何健康问题或医学图像解读，请务必咨询合格的医疗专业人员。")
Window_TITLE = "腹部多器官分割系统 (Synapse)"
# ICON_PATH = "images/UI/abdominal_icon.png" # 可选：替换为腹部器官相关的图标
ICON_PATH = str(UI_ASSET_ROOT / "icon.jpeg") # 暂时沿用旧图标
ZUOZHE = "DA-TransUNet"
WELCOME_LABEL = f'欢迎使用基于深度学习的腹部多器官 ({len(CLASS_NAMES)-1} 类) 分割系统\n'

# 创建必要的目录
UI_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
UI_TMP_ROOT.mkdir(parents=True, exist_ok=True)
UI_RESULT_ROOT.mkdir(parents=True, exist_ok=True) # 保存UI输出结果的目录
LOG_ROOT.mkdir(parents=True, exist_ok=True)
LOGGER = configure_logging()

def model_load(model_path, config_name, num_classes, img_size, device, n_skip=3):
    """
    加载 TransUNet (ViT_seg) 模型
    """
    start_time = time.perf_counter()
    log_event(
        LOGGER,
        "model_load_started",
        "loading model",
        model_path=str(model_path),
        config=config_name,
        device=str(device),
    )
    try:
        config_vit = copy.deepcopy(CONFIGS_ViT_seg.get(config_name))
        if config_vit is None:
            raise ValueError(f"配置 '{config_name}' 不存在于 CONFIGS_ViT_seg 中")

        # 更新配置中的类别数和可能的网格大小
        config_vit.n_classes = num_classes
        config_vit.n_skip = n_skip
        if 'R50' in config_name: # 根据模型名调整 patch grid
             # 假设 vit_patches_size = 16
             vit_patches_size = 16
             config_vit.patches.grid = (img_size // vit_patches_size, img_size // vit_patches_size)

        # 实例化模型
        model = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes)

        # 加载预训练权重
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_event(
            LOGGER,
            "model_load_completed",
            "model loaded",
            duration_ms=duration_ms,
            model_path=str(model_path),
            device=str(device),
        )
        print(f"模型加载成功，耗时 {duration_ms:.0f} ms")
        return model
    except FileNotFoundError:
        log_event(LOGGER, "model_load_failed", "checkpoint not found", level=logging.ERROR, model_path=str(model_path))
        QMessageBox.critical(None, "模型加载错误", f"模型文件未找到于:\n{model_path}\n请检查路径。")
        return None
    except Exception as e:
        log_event(
            LOGGER,
            "model_load_failed",
            str(e),
            level=logging.ERROR,
            model_path=str(model_path),
            exception_type=type(e).__name__,
        )
        QMessageBox.critical(None, "模型加载错误", f"加载模型时发生错误:\n{e}")
        return None

def preprocess_image(img_path, img_size):
    """
    预处理输入图像以适配TransUNet模型
    """
    try:
        image = Image.open(img_path).convert('RGB') # 转换为RGB以匹配模型期望的3通道输入
        image_resized = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
        image_np = np.array(image_resized) / 255.0 # 归一化到 [0, 1]

        # 转置并添加批次维度 HWC -> CHW -> BCHW
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        return image_tensor, np.array(image_resized) # 返回tensor和用于可视化的numpy数组

    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None, None


def get_result(model, img_path, img_size, num_classes, colors, class_names, device):
    """
    获取图像分割结果
    Args:
        model: 已经加载好的模型
        img_path: 需要进行分割的图像路径
        img_size: 模型期望的输入图像大小
        num_classes: 模型输出的类别数
        colors: 用于可视化的颜色列表
        class_names: 类别名称列表
        device: 设备 ('cuda:0' or 'cpu')

    Returns:
        vis_result: 可视化结果 (NumPy array, BGR for cv2)
        organ_info: 各器官占比信息 (string)
    """
    start_time = time.perf_counter()
    log_event(LOGGER, "inference_started", "inference started", image=str(img_path), device=str(device))
    # 预处理图像
    image_tensor, original_resized_np = preprocess_image(img_path, img_size)
    if image_tensor is None:
        log_event(LOGGER, "inference_failed", "preprocess failed", level=logging.ERROR, image=str(img_path))
        return None, "预处理过程中出错。", []

    image_tensor = image_tensor.to(device)

    # 模型推理
    with torch.no_grad():
        output = model(image_tensor) # output shape: [B, num_classes, H, W]
        # 对于某些模型，输出可能在列表或元组中
        if isinstance(output, (list, tuple)):
             output = output[0]
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # shape: [H, W]

    # 创建可视化结果
    vis_result_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    # 计算每个类别的像素数并着色
    total_pixels = pred.shape[0] * pred.shape[1]
    organ_metrics = []

    for class_idx in range(num_classes):
        mask = (pred == class_idx)
        pixel_count = int(np.sum(mask))
        percentage = round((pixel_count / total_pixels) * 100, 2)
        if 0 < class_idx < len(class_names):
            organ_metrics.append(
                {
                    "class_index": class_idx,
                    "name": class_names[class_idx],
                    "percentage": percentage,
                    "color": colors[class_idx] if class_idx < len(colors) else (255, 255, 255),
                }
            )
        if class_idx < len(colors):
             vis_result_rgb[mask] = colors[class_idx]
        else:
             vis_result_rgb[mask] = (255, 255, 255) # 未知类别用白色

    # 将分割结果叠加到原图上
    # original_resized_np 已经是RGB了
    vis_result_overlay = cv2.addWeighted(original_resized_np, 0.6, vis_result_rgb, 0.4, 0)

    # 计算并格式化器官占比信息
    organ_info_lines = ["检测到的器官占比:"]
    # 过滤掉像素数很少的类别，避免噪声影响
    min_pixel_threshold = total_pixels * 0.001 # 阈值：占图像面积的0.1%

    sorted_organs = sorted(organ_metrics, key=lambda item: item["percentage"], reverse=True)

    min_percentage = min_pixel_threshold / total_pixels * 100
    for organ in sorted_organs:
        if organ["percentage"] > min_percentage:
            organ_info_lines.append(f"- {organ['name']}: {organ['percentage']}%")

    if len(organ_info_lines) == 1: # 如果没有检测到显著的器官
        organ_info = "未检测到超过阈值的显著器官。"
    else:
        organ_info = "\n".join(organ_info_lines)

    # 保存结果图像 (BGR格式)
    vis_result_bgr = cv2.cvtColor(vis_result_overlay, cv2.COLOR_RGB2BGR)
    save_path = UI_TMP_ROOT / 'segmentation_result.jpg'
    cv2.imwrite(str(save_path), vis_result_bgr)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    log_event(
        LOGGER,
        "inference_completed",
        "inference completed",
        duration_ms=duration_ms,
        image=str(img_path),
        detected_organs=sum(1 for item in organ_metrics if item["percentage"] > min_percentage),
    )
    print(f"分割结果已保存至: {save_path}，耗时 {duration_ms:.0f} ms")

    return vis_result_bgr, organ_info, organ_metrics


class InferenceWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, model, image_path, device):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.device = device

    def run(self):
        start_time = time.perf_counter()
        try:
            vis_result_bgr, organ_info, organ_metrics = get_result(
                self.model,
                img_path=self.image_path,
                img_size=IMG_SIZE,
                num_classes=NUM_CLASSES,
                colors=COLORS,
                class_names=CLASS_NAMES,
                device=self.device,
            )
            if vis_result_bgr is None:
                raise RuntimeError(organ_info)
            self.completed.emit(
                {
                    "image": vis_result_bgr,
                    "organ_info": organ_info,
                    "organ_metrics": organ_metrics,
                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                }
            )
        except Exception as exc:
            log_event(
                LOGGER,
                "inference_failed",
                str(exc),
                level=logging.ERROR,
                image=str(self.image_path),
                exception_type=type(exc).__name__,
            )
            self.failed.emit(str(exc))


class MainWindow(QTabWidget):
    def __init__(self, model_path, model_config_name=MODEL_CONFIG_NAME, device='auto',
                 model_id=None, n_skip=3):
        super().__init__()
        self.setWindowTitle(Window_TITLE)
        self.resize(1320, 880)
        self.setMinimumSize(1120, 760)
        self.setWindowIcon(QIcon(ICON_PATH))

        self.output_size = 420
        self.device = resolve_torch_device(device)
        self.MODEL_PATH = str(model_path)
        self.current_model_id = model_id
        self.current_config_name = model_config_name
        self.current_n_skip = n_skip
        self.registry_models = []
        self.worker = None
        self.img2predict = ""
        self.last_result = None

        try:
            self.registry_models = list_models()
        except Exception as exc:
            log_event(LOGGER, "registry_load_failed", str(exc), level=logging.ERROR)

        self.seg_model = model_load(
            model_path=self.MODEL_PATH,
            config_name=self.current_config_name,
            num_classes=NUM_CLASSES,
            img_size=IMG_SIZE,
            device=self.device,
            n_skip=self.current_n_skip,
        )
        if self.seg_model is None:
            QMessageBox.critical(self, "严重错误", "加载分割模型失败。应用程序即将关闭。")
            raise SystemExit(1)

        self.initUI()
        self.refresh_system_page()

    def _badge(self, text, state="neutral"):
        label = QLabel(text)
        label.setObjectName("Badge")
        label.setProperty("state", state)
        label.setAlignment(Qt.AlignCenter)
        return label

    def _set_badge(self, badge, text, state):
        badge.setText(text)
        badge.setProperty("state", state)
        badge.style().unpolish(badge)
        badge.style().polish(badge)

    def _panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")
        return panel

    def initUI(self):
        self.setTabPosition(QTabWidget.North)
        self.addTab(self.create_workspace_page(), "工作台")
        self.addTab(self.create_system_page(), "系统监控")
        self.setCurrentIndex(0)
        self.set_stylesheet()

    def create_workspace_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 14)
        title_box = QVBoxLayout()
        title = QLabel("腹部多器官分割工作台")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Synapse CT · TransUNet · 本地推理与结构化分析")
        subtitle.setObjectName("Muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.device_badge = self._badge(f"设备 {str(self.device).upper()}", "info")
        model_name = "未注册模型"
        if self.current_model_id:
            try:
                model_name = get_model(self.current_model_id).display_name
            except KeyError:
                pass
        self.model_badge = self._badge(model_name, "success")
        self.validation_badge = self._badge("技术验证 读取中", "neutral")
        header_layout.addWidget(self.device_badge)
        header_layout.addWidget(self.model_badge)
        header_layout.addWidget(self.validation_badge)
        layout.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(14)
        layout.addLayout(content, 1)

        viewer = self._panel()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(16, 14, 16, 14)
        viewer_layout.setSpacing(12)
        viewer_header = QHBoxLayout()
        viewer_title = QLabel("影像工作区")
        viewer_title.setObjectName("SectionTitle")
        viewer_hint = QLabel("输入切片与彩色分割结果")
        viewer_hint.setObjectName("Muted")
        viewer_header.addWidget(viewer_title)
        viewer_header.addStretch()
        viewer_header.addWidget(viewer_hint)
        viewer_layout.addLayout(viewer_header)

        image_row = QHBoxLayout()
        image_row.setSpacing(12)
        self.left_img = QLabel("等待上传图像")
        self.right_img = QLabel("等待分割结果")
        for label, object_name in [
            (self.left_img, "ImageView"),
            (self.right_img, "ImageView"),
        ]:
            label.setObjectName(object_name)
            label.setMinimumSize(320, 320)
            label.setMaximumSize(470, 470)
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(False)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setWordWrap(True)
        image_row.addWidget(self.left_img, 1)
        image_row.addWidget(self.right_img, 1)
        viewer_layout.addLayout(image_row, 1)

        image_caption = QLabel("左：原始影像    右：模型预测叠加图")
        image_caption.setObjectName("Muted")
        image_caption.setAlignment(Qt.AlignCenter)
        viewer_layout.addWidget(image_caption)
        content.addWidget(viewer, 1)

        analysis = self._panel()
        analysis.setMinimumWidth(340)
        analysis.setMaximumWidth(380)
        analysis_layout = QVBoxLayout(analysis)
        analysis_layout.setContentsMargins(16, 14, 16, 14)
        analysis_layout.setSpacing(10)

        analysis_title = QLabel("分析面板")
        analysis_title.setObjectName("SectionTitle")
        analysis_layout.addWidget(analysis_title)

        model_label = QLabel("模型版本")
        model_label.setObjectName("FieldLabel")
        analysis_layout.addWidget(model_label)
        self.model_selector = QComboBox()
        if self.registry_models:
            for model in self.registry_models:
                self.model_selector.addItem(model.display_name, model.id)
            index = self.model_selector.findData(self.current_model_id)
            if index >= 0:
                self.model_selector.setCurrentIndex(index)
        else:
            self.model_selector.addItem(os.path.basename(self.MODEL_PATH), self.current_model_id)
        self.model_selector.currentIndexChanged.connect(self.switch_model)
        analysis_layout.addWidget(self.model_selector)

        action_row = QHBoxLayout()
        self.upload_button = QPushButton("上传影像")
        self.upload_button.setObjectName("SecondaryButton")
        self.det_img_button = QPushButton("开始分割")
        self.upload_button.clicked.connect(self.upload_img)
        self.det_img_button.clicked.connect(self.detect_img)
        action_row.addWidget(self.upload_button)
        action_row.addWidget(self.det_img_button)
        analysis_layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("等待任务")
        analysis_layout.addWidget(self.progress_bar)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(8)
        self.img_spec_display = QLabel("--")
        self.time_display = QLabel("--")
        self.detected_count_display = QLabel("--")
        for value_label in [self.img_spec_display, self.time_display, self.detected_count_display]:
            value_label.setObjectName("MetricValue")
        metrics.addWidget(QLabel("图像尺寸"), 0, 0)
        metrics.addWidget(self.img_spec_display, 0, 1)
        metrics.addWidget(QLabel("推理耗时"), 1, 0)
        metrics.addWidget(self.time_display, 1, 1)
        metrics.addWidget(QLabel("检出器官"), 2, 0)
        metrics.addWidget(self.detected_count_display, 2, 1)
        analysis_layout.addLayout(metrics)

        organ_title = QLabel("器官分布")
        organ_title.setObjectName("FieldLabel")
        analysis_layout.addWidget(organ_title)
        self.organ_table = QTableWidget(0, 3)
        self.organ_table.setHorizontalHeaderLabels(["器官", "占比", "状态"])
        self.organ_table.verticalHeader().setVisible(False)
        self.organ_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.organ_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.organ_table.setShowGrid(False)
        self.organ_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.organ_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.organ_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.organ_table.setMinimumHeight(245)
        analysis_layout.addWidget(self.organ_table)

        self.result_summary = QLabel("上传影像后开始分割，结果将在此处汇总。")
        self.result_summary.setObjectName("SummaryBox")
        self.result_summary.setWordWrap(True)
        self.result_summary.setMinimumHeight(64)
        analysis_layout.addWidget(self.result_summary)

        self.export_button = QPushButton("导出分析报告")
        self.export_button.setObjectName("SecondaryButton")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_report)
        analysis_layout.addWidget(self.export_button)
        analysis_layout.addStretch()
        content.addWidget(analysis)
        return page

    def create_system_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("系统监控与验证")
        title.setObjectName("PageTitle")
        subtitle = QLabel("模型注册、运行日志、技术验证与环境状态")
        subtitle.setObjectName("Muted")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.stat_device = self._stat_tile("运行设备", "--")
        self.stat_models = self._stat_tile("可用模型", "--")
        self.stat_latency = self._stat_tile("平均推理耗时", "--")
        self.stat_errors = self._stat_tile("运行错误", "--")
        for tile in [self.stat_device, self.stat_models, self.stat_latency, self.stat_errors]:
            stats.addWidget(tile)
        layout.addLayout(stats)

        model_panel = self._panel()
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(16, 14, 16, 14)
        model_title = QLabel("模型版本注册表")
        model_title.setObjectName("SectionTitle")
        model_layout.addWidget(model_title)
        self.model_table = QTableWidget(0, 4)
        self.model_table.setHorizontalHeaderLabels(["模型", "状态", "Dice", "检查点"])
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.model_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.model_table.setMaximumHeight(150)
        model_layout.addWidget(self.model_table)
        layout.addWidget(model_panel)

        lower = QHBoxLayout()
        lower.setSpacing(12)
        validation_panel = self._panel()
        validation_layout = QVBoxLayout(validation_panel)
        validation_layout.setContentsMargins(16, 14, 16, 14)
        validation_title = QLabel("技术验证")
        validation_title.setObjectName("SectionTitle")
        validation_layout.addWidget(validation_title)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMinimumHeight(170)
        validation_layout.addWidget(self.validation_text)
        validation_actions = QHBoxLayout()
        self.refresh_button = QPushButton("刷新状态")
        self.refresh_button.setObjectName("SecondaryButton")
        self.open_report_button = QPushButton("打开验证报告")
        self.refresh_button.clicked.connect(self.refresh_system_page)
        self.open_report_button.clicked.connect(self.open_validation_report)
        validation_actions.addWidget(self.refresh_button)
        validation_actions.addWidget(self.open_report_button)
        validation_layout.addLayout(validation_actions)
        lower.addWidget(validation_panel, 1)

        log_panel = self._panel()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(16, 14, 16, 14)
        log_title = QLabel("最近运行日志")
        log_title.setObjectName("SectionTitle")
        log_layout.addWidget(log_title)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(170)
        log_layout.addWidget(self.log_view)
        lower.addWidget(log_panel, 1)
        layout.addLayout(lower, 1)

        footer = QLabel(
            f"{ZUOZHE} · 仅用于科研、教学与算法验证，不构成临床诊断或医疗器械建议。"
        )
        footer.setObjectName("Muted")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        return page

    def _stat_tile(self, title, value):
        tile = QFrame()
        tile.setObjectName("StatTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(14, 10, 14, 10)
        title_label = QLabel(title)
        title_label.setObjectName("Muted")
        value_label = QLabel(value)
        value_label.setObjectName("StatValue")
        tile_layout.addWidget(title_label)
        tile_layout.addWidget(value_label)
        tile.value_label = value_label
        return tile

    def set_stylesheet(self):
        style = """
        * {
            font-family: 'Microsoft YaHei UI', 'Microsoft YaHei';
        }
        QTabWidget::pane {
            border: none;
            background: #f4f7f9;
        }
        QTabBar::tab {
            background: #e8eef3;
            color: #48606f;
            border: 1px solid #d7e0e7;
            border-radius: 7px 7px 0 0;
            min-width: 132px;
            min-height: 28px;
            padding: 7px 18px;
            margin-right: 6px;
            font-size: 14px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #0f766e;
            border-color: #c8d7df;
            border-bottom-color: #ffffff;
            font-weight: 600;
        }
        QTabBar::tab:hover:!selected {
            background: #f1f5f8;
        }
        QWidget {
            color: #1f2937;
            font-size: 12px;
        }
        QFrame#Header, QFrame#Panel, QFrame#StatTile {
            background: #ffffff;
            border: 1px solid #dbe4ea;
            border-radius: 8px;
        }
        QLabel#PageTitle {
            font-size: 23px;
            font-weight: 700;
            color: #17252f;
        }
        QLabel#SectionTitle {
            font-size: 15px;
            font-weight: 700;
            color: #243746;
        }
        QLabel#Muted {
            color: #6b7f8c;
            font-size: 11px;
        }
        QLabel#FieldLabel {
            color: #4b6472;
            font-weight: 600;
        }
        QLabel#MetricValue {
            color: #0f766e;
            font-weight: 700;
            font-size: 14px;
        }
        QLabel#Badge {
            border-radius: 10px;
            padding: 5px 10px;
            font-size: 11px;
            font-weight: 600;
        }
        QLabel#Badge[state="success"] {
            background: #e8f5f1;
            color: #0f766e;
            border: 1px solid #b7ded4;
        }
        QLabel#Badge[state="info"] {
            background: #eef4fa;
            color: #275f8b;
            border: 1px solid #c7d9e8;
        }
        QLabel#Badge[state="warning"] {
            background: #fff5e8;
            color: #a35a09;
            border: 1px solid #f0d1a4;
        }
        QLabel#Badge[state="error"] {
            background: #fff0f0;
            color: #b42318;
            border: 1px solid #efc1bd;
        }
        QLabel#Badge[state="neutral"] {
            background: #f1f4f6;
            color: #61727d;
            border: 1px solid #d8e0e5;
        }
        QLabel#ImageView {
            background: #0f1519;
            color: #a9bac5;
            border: 1px solid #26343d;
            border-radius: 6px;
            font-size: 15px;
        }
        QLabel#SummaryBox {
            background: #f6f9fb;
            border: 1px solid #d9e3e9;
            border-radius: 6px;
            padding: 10px;
            color: #415664;
        }
        QLabel#StatValue {
            color: #17252f;
            font-size: 20px;
            font-weight: 700;
        }
        QPushButton {
            background: #0f766e;
            border: none;
            border-radius: 6px;
            color: white;
            padding: 9px 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #0d655e;
        }
        QPushButton:pressed {
            background: #0a544e;
        }
        QPushButton:disabled {
            background: #c7d2d8;
            color: #7d8d96;
        }
        QPushButton#SecondaryButton {
            background: #eef4f6;
            color: #245364;
            border: 1px solid #cddbe2;
        }
        QPushButton#SecondaryButton:hover {
            background: #e2edf1;
        }
        QComboBox {
            background: #f8fafb;
            border: 1px solid #cedbe3;
            border-radius: 6px;
            padding: 7px 10px;
            color: #243746;
        }
        QProgressBar {
            border: 1px solid #d4e0e6;
            border-radius: 6px;
            background: #f5f8fa;
            text-align: center;
            height: 20px;
            color: #36515f;
        }
        QProgressBar::chunk {
            background: #14b8a6;
            border-radius: 5px;
        }
        QTableWidget {
            background: #ffffff;
            border: none;
            alternate-background-color: #f8fafb;
            selection-background-color: #e8f5f1;
            gridline-color: transparent;
        }
        QHeaderView::section {
            background: #f2f6f8;
            color: #536a78;
            border: none;
            border-bottom: 1px solid #dae4ea;
            padding: 7px;
            font-weight: 600;
        }
        QTextEdit {
            background: #f8fafb;
            border: 1px solid #d7e2e8;
            border-radius: 6px;
            padding: 8px;
            color: #405560;
        }
        QScrollBar:vertical {
            width: 9px;
            background: transparent;
        }
        QScrollBar::handle:vertical {
            background: #c4d2da;
            border-radius: 4px;
            min-height: 24px;
        }
        """
        self.setStyleSheet(style)

    def switch_model(self, index):
        if index < 0 or not self.registry_models:
            return
        model_id = self.model_selector.itemData(index)
        if not model_id or model_id == self.current_model_id:
            return
        try:
            model = get_model(model_id)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            loaded = model_load(
                model_path=str(model.checkpoint),
                config_name=model.config,
                num_classes=model.num_classes,
                img_size=model.img_size,
                device=self.device,
                n_skip=model.n_skip,
            )
            if loaded is None:
                raise RuntimeError("模型加载失败")
            self.seg_model = loaded
            self.MODEL_PATH = str(model.checkpoint)
            self.current_model_id = model.id
            self.current_config_name = model.config
            self.current_n_skip = model.n_skip
            self._set_badge(self.model_badge, model.display_name, "success")
            self.result_summary.setText("模型已切换，请重新执行分割。")
            self.export_button.setEnabled(False)
            self.last_result = None
        except Exception as exc:
            QMessageBox.critical(self, "模型切换失败", str(exc))
            previous = self.model_selector.findData(self.current_model_id)
            if previous >= 0:
                self.model_selector.blockSignals(True)
                self.model_selector.setCurrentIndex(previous)
                self.model_selector.blockSignals(False)
        finally:
            QApplication.restoreOverrideCursor()

    def display_image(self, label_widget, image_path_or_array):
        if isinstance(image_path_or_array, str): # 如果是路径
             # 检查路径是否存在
             if not os.path.exists(image_path_or_array):
                 print(f"警告：图像路径不存在 {image_path_or_array}")
                 label_widget.setText("图像文件丢失")
                 return
             pixmap = QPixmap(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray): # 如果是Numpy数组 (BGR)
             if image_path_or_array.size == 0:
                 print("警告：尝试显示空图像数组。")
                 label_widget.setText("无效图像")
                 return
             try:
                 if image_path_or_array.dtype != np.uint8:
                     if image_path_or_array.max() <= 1.0 and image_path_or_array.min() >=0:
                         image_path_or_array = (image_path_or_array * 255).astype(np.uint8)
                     else:
                         image_path_or_array = image_path_or_array.astype(np.uint8)

                 if len(image_path_or_array.shape) == 3 and image_path_or_array.shape[2] == 3:
                     height, width, channel = image_path_or_array.shape
                     bytesPerLine = 3 * width
                     qImg = QImage(image_path_or_array.data, width, height, bytesPerLine, QImage.Format_BGR888)
                 elif len(image_path_or_array.shape) == 2:
                     height, width = image_path_or_array.shape
                     bytesPerLine = width
                     qImg = QImage(image_path_or_array.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
                 else:
                      print(f"警告：不支持的图像形状 {image_path_or_array.shape}")
                      label_widget.setText("不支持的图像格式")
                      return
                 pixmap = QPixmap.fromImage(qImg)
             except Exception as e:
                 print(f"从Numpy数组创建QPixmap时出错: {e}")
                 label_widget.setText("显示图像时出错")
                 return
        else:
            label_widget.setText("无效的图像来源")
            return

        # 缩放并显示，使用 self.output_size
        scaled_pixmap = pixmap.scaled(self.output_size, self.output_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)
        label_widget.setText("") # 清除可能存在的文本

    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            '选择医学影像切片',
            '',
            '图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)',
        )
        if fileName:
            try:
                if not os.path.exists(fileName):
                    QMessageBox.warning(self, "文件错误", f"文件未找到: {fileName}")
                    return

                suffix = os.path.splitext(fileName)[1] or ".png"
                save_path = str(UI_TMP_ROOT / ("tmp_upload" + suffix))
                shutil.copy(fileName, save_path)
                self.img2predict = save_path

                self.display_image(self.left_img, self.img2predict)
                self.right_img.setPixmap(QPixmap())
                self.right_img.setText("等待分割结果")
                self.img_spec_display.setText("--")
                self.time_display.setText("--")
                self.detected_count_display.setText("--")
                self.organ_table.setRowCount(0)
                self.result_summary.setText("影像已载入，可以开始分割。")
                self.progress_bar.setValue(0)
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setFormat("等待任务")
                self.export_button.setEnabled(False)
                self.last_result = None
                log_event(LOGGER, "image_uploaded", "image uploaded", source=fileName, stored=save_path)

            except Exception as e:
                QMessageBox.critical(self, "上传失败", f"图片上传失败: {str(e)}")
                log_event(LOGGER, "image_upload_failed", str(e), level=logging.ERROR, source=fileName)
                self.img2predict = ""


    def detect_img(self):
        if not self.img2predict or not os.path.exists(self.img2predict):
            QMessageBox.warning(self, "警告", "请先上传有效的图片。")
            return

        if self.seg_model is None:
            QMessageBox.critical(self, "错误", "分割模型未加载。")
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "任务进行中", "当前分割任务尚未完成。")
            return

        self.upload_button.setEnabled(False)
        self.det_img_button.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("正在推理")
        self.result_summary.setText("模型正在分析影像，请稍候。")

        try:
            with Image.open(self.img2predict) as image:
                self.img_spec_display.setText(f"{image.width} × {image.height}")
        except Exception:
            self.img_spec_display.setText("未知")

        self.worker = InferenceWorker(self.seg_model, self.img2predict, self.device)
        self.worker.completed.connect(self.on_inference_completed)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_inference_completed(self, payload):
        self.display_image(self.right_img, payload["image"])
        self.time_display.setText(f"{payload['elapsed_seconds']:.3f} 秒")
        self.populate_organ_table(payload["organ_metrics"])
        detected = sum(1 for item in payload["organ_metrics"] if item["percentage"] > 0.1)
        self.detected_count_display.setText(f"{detected} / {len(payload['organ_metrics'])}")
        self.result_summary.setText(
            f"已完成推理，检出 {detected} 个显著器官。结果仅用于科研、教学与算法验证。"
        )
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("分析完成")
        self.last_result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_id": self.current_model_id,
            "model_path": self.MODEL_PATH,
            "input_image": self.img2predict,
            "elapsed_seconds": payload["elapsed_seconds"],
            "organ_metrics": payload["organ_metrics"],
            "organ_info": payload["organ_info"],
            "device": str(self.device),
        }
        self.export_button.setEnabled(True)
        print(payload["organ_info"])

    def on_inference_failed(self, message):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("任务失败")
        self.result_summary.setText("分割失败，请检查日志或模型配置。")
        QMessageBox.critical(self, "检测失败", f"分割过程中发生错误:\n{message}")
        traceback.print_exc()

    def on_worker_finished(self):
        self.upload_button.setEnabled(True)
        self.det_img_button.setEnabled(True)
        self.model_selector.setEnabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def populate_organ_table(self, organ_metrics):
        self.organ_table.setRowCount(0)
        for organ in sorted(organ_metrics, key=lambda item: item["percentage"], reverse=True):
            row = self.organ_table.rowCount()
            self.organ_table.insertRow(row)
            color = QColor(*organ["color"])
            name_item = QTableWidgetItem("●  " + organ["name"])
            name_item.setForeground(color)
            self.organ_table.setItem(row, 0, name_item)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(min(100, int(round(organ["percentage"]))))
            progress.setFormat(f"{organ['percentage']:.2f}%")
            progress.setStyleSheet(
                f"QProgressBar {{ border: none; background: #eef2f4; height: 18px; color: #304957; }}"
                f"QProgressBar::chunk {{ background: {color.name()}; border-radius: 4px; }}"
            )
            self.organ_table.setCellWidget(row, 1, progress)

            status = "检出" if organ["percentage"] > 0.1 else "未检出"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.organ_table.setItem(row, 2, status_item)
            self.organ_table.setRowHeight(row, 29)

    def refresh_system_page(self):
        try:
            health = get_system_health()
        except Exception as exc:
            log_event(LOGGER, "monitoring_failed", str(exc), level=logging.ERROR)
            return

        self.stat_device.value_label.setText(
            "CUDA" if health["cuda_available"] else "CPU"
        )
        self.stat_models.value_label.setText(str(health["registered_models"]))
        average = health["logging"].get("average_inference_duration_ms")
        self.stat_latency.value_label.setText(f"{average:.0f} ms" if average else "--")
        self.stat_errors.value_label.setText(str(health["logging"].get("error_count", 0)))

        self.model_table.setRowCount(0)
        for model in self.registry_models:
            row = self.model_table.rowCount()
            self.model_table.insertRow(row)
            self.model_table.setItem(row, 0, QTableWidgetItem(model.display_name))
            self.model_table.setItem(row, 1, QTableWidgetItem(model.status))
            dice = model.metrics.get("mean_dice")
            self.model_table.setItem(row, 2, QTableWidgetItem(f"{dice:.4f}" if dice else "--"))
            self.model_table.setItem(row, 3, QTableWidgetItem(str(model.checkpoint)))
            self.model_table.setRowHeight(row, 30)

        validation = health.get("validation", {})
        if validation.get("available"):
            state = "通过" if validation.get("passed") else "未通过"
            self.validation_text.setPlainText(
                "状态：技术回顾性验证{}\n"
                "平均 Dice：{:.4f}\n"
                "平均 HD95：{:.2f}\n"
                "临床验证：未开展\n\n"
                "说明：该结果来自公开数据集的离线评估，不构成临床验证或诊断依据。".format(
                    state,
                    validation.get("mean_dice") or 0.0,
                    validation.get("mean_hd95") or 0.0,
                )
            )
            self._set_badge(
                self.validation_badge,
                "技术验证 通过" if validation.get("passed") else "技术验证 未通过",
                "success" if validation.get("passed") else "warning",
            )
        else:
            self.validation_text.setPlainText("尚未生成技术验证报告。")
            self._set_badge(self.validation_badge, "技术验证 缺失", "warning")

        events = health["logging"].get("recent_events", [])
        log_lines = []
        for event in reversed(events):
            timestamp = event.get("timestamp", "")[11:19]
            fields = event.get("fields", {})
            duration = fields.get("duration_ms")
            duration_text = f" · {duration:.0f} ms" if isinstance(duration, (int, float)) else ""
            log_lines.append(
                f"{timestamp}  {event.get('event', 'unknown')}{duration_text}\n{event.get('message', '')}"
            )
        self.log_view.setPlainText("\n\n".join(log_lines) if log_lines else "暂无运行日志。")
        self.open_report_button.setEnabled(DEFAULT_REPORT_HTML.exists())

    def export_report(self):
        if not self.last_result:
            QMessageBox.warning(self, "无可导出结果", "请先完成一次分割。")
            return

        default_name = UI_RESULT_ROOT / ("analysis_" + time.strftime("%Y%m%d_%H%M%S") + ".pdf")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "导出分析报告",
            str(default_name),
            "PDF 报告 (*.pdf);;TXT 报告 (*.txt);;JSON 报告 (*.json);;HTML 报告 (*.html)",
        )
        if not file_name:
            return

        report = dict(self.last_result)
        report["validation"] = load_latest_report()
        report["disclaimer"] = "仅用于科研、教学与算法验证，不构成临床诊断。"
        try:
            exported_path = export_analysis_report(report, file_name)
        except Exception as exc:
            log_event(LOGGER, "report_export_failed", str(exc), level=logging.ERROR, path=file_name)
            QMessageBox.critical(self, "导出失败", f"无法导出报告:\n{exc}")
            return
        log_event(
            LOGGER,
            "report_exported",
            "analysis report exported",
            path=str(exported_path),
            format=exported_path.suffix.lower(),
        )
        QMessageBox.information(self, "导出完成", f"分析报告已保存至:\n{exported_path}")

    def open_validation_report(self):
        if not DEFAULT_REPORT_HTML.exists():
            QMessageBox.warning(self, "报告不存在", "请先生成技术验证报告。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DEFAULT_REPORT_HTML)))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "任务进行中", "请等待当前分割任务完成后再退出。")
            event.ignore()
            return
        reply = QMessageBox.question(self,
                                    '退出确认',
                                    "您确定要退出应用程序吗？",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            log_event(LOGGER, "app_closed", "application closed")
            try:
                if UI_TMP_ROOT.exists():
                    shutil.rmtree(UI_TMP_ROOT)
            except Exception as e:
                log_event(LOGGER, "cleanup_failed", str(e), level=logging.ERROR)
            event.accept()
        else:
            event.ignore()

def parse_args():
    parser = argparse.ArgumentParser(description="Synapse abdominal multi-organ segmentation UI")
    parser.add_argument("--model", type=str, default=None,
                        help="path to the PyTorch checkpoint")
    parser.add_argument("--model-id", type=str, default=None,
                        help="registered model id; defaults to the first registered model")
    parser.add_argument("--config", type=str, default=None,
                        help="network configuration in CONFIGS_ViT_seg")
    parser.add_argument("--n-skip", type=int, default=None,
                        help="number of decoder skip connections")
    parser.add_argument("--device", type=str, default="auto",
                        help="runtime device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--smoke-test-image", type=str, default=None,
                        help="run one inference, save a UI screenshot, then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_model = None
    try:
        registered_models = list_models()
    except Exception:
        registered_models = []

    if args.model_id:
        selected_model = get_model(args.model_id)
    elif args.model is None and registered_models:
        selected_model = registered_models[0]

    if selected_model is not None:
        model_path = selected_model.checkpoint
        config_name = args.config or selected_model.config
        model_id = selected_model.id
        n_skip = args.n_skip if args.n_skip is not None else selected_model.n_skip
    else:
        model_path = resolve_path(args.model or DEFAULT_CHECKPOINT)
        config_name = args.config or MODEL_CONFIG_NAME
        model_id = None
        n_skip = args.n_skip if args.n_skip is not None else 3

    app = QApplication(sys.argv)

    if not model_path.exists():
        print(f"警告：默认模型路径 '{model_path}' 不存在。")
        model_file, _ = QFileDialog.getOpenFileName(
            None, '选择分割模型文件', str(model_path.parent), 'PyTorch 模型文件 (*.pth *.pt)'
        )
        if not model_file or not os.path.exists(model_file):
            QMessageBox.critical(
                None,
                "未找到模型",
                f"未选择有效的模型文件或在默认路径未找到模型:\n{model_path}\n应用程序无法启动。",
            )
            sys.exit(1)
        model_path = resolve_path(model_file)
        config_name = args.config or MODEL_CONFIG_NAME
        print(f"用户选择的模型路径: {model_path}")
    else:
        print(f"使用默认模型路径: {model_path}")

    mainWindow = MainWindow(
        model_path=str(model_path),
        model_config_name=config_name,
        device=args.device,
        model_id=model_id,
        n_skip=n_skip,
    )

    if args.smoke_test_image:
        smoke_image = resolve_path(args.smoke_test_image)
        if not smoke_image.exists():
            raise FileNotFoundError(f"Smoke-test image not found: {smoke_image}")

        start_time = time.time()
        vis_result_bgr, organ_info, organ_metrics = get_result(
            mainWindow.seg_model,
            img_path=str(smoke_image),
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            colors=COLORS,
            class_names=CLASS_NAMES,
            device=mainWindow.device,
        )
        if vis_result_bgr is None:
            raise RuntimeError("Smoke-test inference failed")

        mainWindow.display_image(mainWindow.left_img, str(smoke_image))
        with Image.open(smoke_image) as image:
            mainWindow.img_spec_display.setText(f"{image.width} × {image.height}")
        mainWindow.on_inference_completed(
            {
                "image": vis_result_bgr,
                "organ_info": organ_info,
                "organ_metrics": organ_metrics,
                "elapsed_seconds": round(time.time() - start_time, 3),
            }
        )
        mainWindow.show()
        app.processEvents()

        screenshot_path = UI_RESULT_ROOT / "ui_smoke_result.png"
        mainWindow.grab().save(str(screenshot_path))
        mainWindow.setCurrentIndex(1)
        app.processEvents()
        system_screenshot_path = UI_RESULT_ROOT / "ui_system_result.png"
        mainWindow.grab().save(str(system_screenshot_path))
        print("SMOKE_TEST_OK")
        print(organ_info)
        print(f"UI screenshot: {screenshot_path}")
        print(f"System screenshot: {system_screenshot_path}")
        print(f"Segmentation image: {UI_TMP_ROOT / 'segmentation_result.jpg'}")
        mainWindow.deleteLater()
        app.quit()
        sys.exit(0)

    mainWindow.show()
    sys.exit(app.exec())
