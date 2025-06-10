"""
Test Client for Crop Disease Classification API

This script demonstrates how to interact with the FastAPI server
from IoT devices like Raspberry Pi.
"""

import requests
import json
from pathlib import Path
import time

class CropDiseaseClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check if the API server is running and healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Server is healthy!")
                print(f"   Model loaded: {data['model_loaded']}")
                print(f"   Device: {data['device']}")
                print(f"   Supported formats: {', '.join(data['supported_formats'])}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def get_classes(self):
        """Get all available crop disease classes."""
        try:
            response = self.session.get(f"{self.base_url}/classes")
            if response.status_code == 200:
                data = response.json()
                print(f"📋 Available classes: {data['total_classes']}")
                for class_info in data['classes'][:5]:  # Show first 5
                    print(f"   {class_info['crop']} - {class_info['disease']}")
                if data['total_classes'] > 5:
                    print(f"   ... and {data['total_classes'] - 5} more")
                return data
            else:
                print(f"❌ Failed to get classes: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting classes: {e}")
            return None
    
    def predict_image(self, image_path):
        """
        Predict crop disease from an image file.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            dict: Prediction results or None if failed
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                print(f"❌ Image file not found: {image_path}")
                return None
            
            print(f"🔍 Analyzing image: {image_path.name}")
            
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction successful!")
                print(f"   🌱 Crop: {result['crop']}")
                print(f"   🦠 Disease: {result['disease']}")
                print(f"   📊 Confidence: {result['confidence']:.2%}")
                print(f"   ⏱️  Processing time: {result['model_info']['processing_time_seconds']}s")
                return result
            else:
                error_data = response.json()
                print(f"❌ Prediction failed: {error_data.get('message', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making prediction: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def predict_batch(self, image_paths):
        """
        Predict crop disease for multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            dict: Batch prediction results or None if failed
        """
        try:
            files = []
            for image_path in image_paths:
                image_path = Path(image_path)
                if image_path.exists():
                    files.append(('files', (image_path.name, open(image_path, 'rb'), 'image/jpeg')))
                else:
                    print(f"⚠️  Skipping missing file: {image_path}")
            
            if not files:
                print("❌ No valid image files found")
                return None
            
            print(f"🔍 Analyzing {len(files)} images...")
            
            response = self.session.post(f"{self.base_url}/predict/batch", files=files)
            
            # Close file handles
            for _, (_, file_handle, _) in files:
                file_handle.close()
            
            if response.status_code == 200:
                results = response.json()
                print("✅ Batch prediction completed!")
                for result in results['batch_results']:
                    if 'prediction' in result:
                        pred = result['prediction']
                        print(f"   📁 {result['filename']}: {pred['crop']} - {pred['disease']} ({pred['confidence']:.2%})")
                    else:
                        print(f"   ❌ {result['filename']}: {result.get('error', 'Unknown error')}")
                return results
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making batch prediction: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

def main():
    """Test the API client."""
    print("🚀 Crop Disease Classification API Test Client")
    print("=" * 50)
    
    # Initialize client
    client = CropDiseaseClient()
    
    # Health check
    print("\n1. Health Check")
    if not client.health_check():
        print("Cannot proceed without a healthy server. Please start the API server first.")
        return
    
    # Get classes
    print("\n2. Available Classes")
    client.get_classes()
    
    # Example prediction (you can replace with actual image paths)
    print("\n3. Single Image Prediction")
    print("To test image prediction, place an image in the current directory")
    print("and update the image path in the script.")
    
    # Uncomment and modify these lines to test with actual images:
    # result = client.predict_image("path/to/your/test_image.jpg")
    
    print("\n✅ Test completed!")
    print("\n📖 API Documentation available at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
