import cv2
import numpy as np
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

from app.utils.model_utils import load_saved_model_weights


class DiseasePredictionPipeline:
    def __init__(self, detection_method="yolo", confidence_threshold=0.5):
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        self.leaf_detector = None
        self.crop_classifiers = {}
        self.setup_leaf_detection()

    def setup_leaf_detection(self):
        if self.detection_method == "yolo":
            self.setup_yolo_detection()
        else:
            print(f"⚠️ Unknown detection method: {self.detection_method}")
            print("🔄 Falling back to traditional computer vision...")

    def setup_yolo_detection(self):
        """Setup YOLOv8 for leaf detection"""
        try:
            print("📦 Setting up YOLOv8 leaf detection...")
            # Try to import and setup YOLOv8
            model_path = "./app/models/weights/yolov8n.pt"
            self.leaf_detector = YOLO(model_path)

            self.leaf_detector.conf = self.confidence_threshold  # confidence threshold
            self.leaf_detector.iou = 0.45

            print(f"✅ YOLOv8 detector ready (loaded from {model_path})")

        except Exception as e:
            print(f"❌ YOLOv8 setup failed: {e}")
            print("🔄 Falling back to traditional detection...")

    def detect_leaves_in_image(self, image):
        """Detect leaves in a real-world image"""
        print(f"🔍 Detecting leaves using {self.detection_method} method...")

        if self.detection_method == "yolo":
            # Run YOLO prediction
            results = self.leaf_detector(image)

            # Process results into our standard format
            detections = []
            for result in results:
                for box in result.boxes:
                    # Convert tensor to list and get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    cls_id = box.cls.item()
                    cls_name = result.names[cls_id]

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class": cls_name,
                            "class_id": cls_id,
                        }
                    )

            return detections
        else:
            return []

    def load_crop_classifier(self, model_type="ultra_quick"):
        """Load a trained crop disease classifier"""
        try:
            model, class_info = load_saved_model_weights(model_type)
            if model is not None:
                self.crop_classifiers = {
                    "model": model,
                    "class_info": class_info,
                    "classes": class_info["unified_classes"],
                    "class_indices": class_info["model_params"],
                }
                print(f"✅ Loaded classifier ({model_type})")
                return True
            else:
                print("❌ Failed to load classifier")
                return False
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            return False

    def preprocess_leaf_for_classification(self, image, bbox):
        """Extract and preprocess leaf region for disease classification"""
        x1, y1, x2, y2 = bbox

        # Extract leaf region
        leaf_region = image[y1:y2, x1:x2]

        if leaf_region.size == 0:
            return None

        # Resize to model input size (224x224)
        leaf_resized = cv2.resize(leaf_region, (224, 224))

        # Convert BGR to RGB
        leaf_rgb = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        leaf_normalized = leaf_rgb.astype(np.float32) / 255.0

        # Add batch dimension
        leaf_batch = np.expand_dims(leaf_normalized, axis=0)

        return leaf_batch

    def predict_leaf_disease(self, leaf_image):
        """Classify disease in a detected leaf"""

        try:
            classifier = self.crop_classifiers
            model = classifier["model"]
            classes = classifier["classes"]

            # Get prediction
            predictions = model.predict(leaf_image, verbose=0)

            # Get top prediction
            top_idx = np.argmax(predictions[0])
            confidence = predictions[0][top_idx]
            predicted_class = classes[top_idx]

            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = []

            for idx in top3_indices:
                top3_predictions.append(
                    {"class": classes[idx], "confidence": float(predictions[0][idx])}
                )

            return predicted_class, float(confidence), top3_predictions

        except Exception as e:
            return None, 0.0, f"Classification error: {e}"
