import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
from typing import List, Tuple, Dict
import config

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
    def _load_class_names(self) -> List[str]:
        """Load class names from JSON file"""
        class_names_path = os.path.join(config.MODEL_DIR, "class_names.json")
        try:
            with open(class_names_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Class names file not found at {class_names_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {class_names_path}")

    def _load_model(self) -> torch.nn.Module:
        """Load trained model weights"""
        try:
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
            
            # Load state dict with proper device mapping
            state_dict = torch.load(config.MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def _get_transforms(self):
        """Get image transformations matching training setup"""
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path: str) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
            - predicted_class: Name of the predicted class
            - confidence: Confidence score (0-100)
            - top_predictions: List of top predictions with scores
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                
            # Get top predictions
            top_probs, top_classes = torch.topk(probabilities, k=3)
            
            return {
                "predicted_class": self.class_names[top_classes[0].item()],
                "confidence": top_probs[0].item(),
                "top_predictions": [
                    (self.class_names[cls.item()], prob.item())
                    for cls, prob in zip(top_classes, top_probs)
                ],
                "success": True
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Image file not found at {image_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing image: {str(e)}"
            }

# Example usage
if __name__ == "__main__":
    classifier = ImageClassifier()
    
    # Test with an image
    test_image = "path/to/your/test_image.jpg"  # Replace with your image path
    result = classifier.predict_image(test_image)
    
    if result["success"]:
        print("\nPrediction Results:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nTop Predictions:")
        for i, (cls, conf) in enumerate(result['top_predictions'], 1):
            print(f"{i}. {cls}: {conf:.2f}%")
    else:
        print(f"Error: {result['error']}")