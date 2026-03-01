"""
YOLO Detection Engine.
Handles model loading, inference, and result processing.
"""
import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import MODELS_DIR, AVAILABLE_MODELS


class YOLODetector:
    """Wrapper class for YOLO model operations with caching."""

    def __init__(self, model_name: str = "YOLOv8n (Nano)"):
        """
        Initialize YOLO detector.

        Args:
            model_name: Name of the YOLO model to use
        """
        self.model_name = model_name
        self.model_file = AVAILABLE_MODELS[model_name]["file"]
        self.model = self._load_model(self.model_file)
        self.device = self._get_device()

    @staticmethod
    @st.cache_resource(show_spinner="Loading YOLO model...")
    def _load_model(model_file: str) -> YOLO:
        """
        Load YOLO model with caching.

        Args:
            model_file: Name of the model file

        Returns:
            Loaded YOLO model
        """
        try:
            model_path = MODELS_DIR / model_file

            # Model will auto-download if not exists
            model = YOLO(str(model_path))

            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise

    def _get_device(self) -> str:
        """
        Detect available device (GPU/CPU).

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
        else:
            return 'cpu'

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640
    ) -> List[Dict]:
        """
        Run object detection on an image.

        Args:
            image: Input image (BGR format from OpenCV)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            image_size: Input image size for model

        Returns:
            List of detection dictionaries with keys:
            'box', 'class_id', 'class_name', 'confidence'
        """
        try:
            # Run inference
            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                device=self.device,
                verbose=False
            )

            # Process results
            detections = []
            if len(results) > 0:
                result = results[0]

                # Extract boxes, classes, and confidences
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    # Get class names
                    names = result.names

                    # Format detections
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        detections.append({
                            'box': box.tolist(),
                            'class_id': int(cls_id),
                            'class_name': names[cls_id],
                            'confidence': float(conf)
                        })

            return detections

        except Exception as e:
            st.error(f"Detection error: {e}")
            return []

    def get_model_info(self) -> Dict:
        """
        Get information about the current model.

        Returns:
            Dictionary with model metadata
        """
        model_info = AVAILABLE_MODELS[self.model_name].copy()
        model_info['device'] = self.device
        model_info['model_name'] = self.model_name
        return model_info

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available model names.

        Returns:
            List of model names
        """
        return list(AVAILABLE_MODELS.keys())
