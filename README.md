# 🌿 Crop Doctor ## Overview
Crop Doctor AI is a smart, image-based diagnosis system designed to help farmers and agricultural experts quickly identify plant diseases from leaf images. The system simulates AI-powered predictions through a web-based interface, providing instant feedback in a simple, user-friendly design.

This project addresses the challenge of limited access to plant health diagnostics in rural and remote areas, offering a scalable solution for rapid disease detection.

---

## Objectives
- Provide an accessible web tool for quick plant disease identification.
- Minimize diagnosis time by delivering results instantly.
- Build a modular backend ready for future integration with real AI models trained on datasets such as *PlantVillage*.

---

## Key Features
- **Image Upload Interface** – Users can upload crop leaf photos from any device.
- **Instant Predictions** – Generates a possible disease label in seconds.
- **Extensible Design** – Ready for integration with actual deep learning models.
- **Simple User Experience** – Optimized for both desktop and mobile devices.

---

## System Workflow
1. **Image Acquisition** – User uploads a leaf image.
2. **Prediction Engine** – The backend processes the image and generates a simulated prediction.
3. **Results Display** – The predicted disease name is displayed on the interface.

---

## Technology Stack
- **Frontend:** HTML, CSS
- **Backend:** Python (Flask)
- **Prediction Logic:** Simulated AI Model (rule-based/randomized for prototype)
- **Deployment Ready:** Cloud-friendly structure for future deployment

---

## Development Steps (Code Order)
This is the exact order in which the project was built:

1. **Project Folder Setup**
```

crop-doctor-ai/
├── app.py
├── requirements.txt
├── utils/
├── templates/
├── static/

````

2. **Dependency Definition**
- Created `requirements.txt` with:
  ```txt
  flask
  ```

3. **Prediction Logic**
- Added `utils/predict.py` containing a simulated AI model using random predictions:
  ```python
  import random

  DISEASES = [
      "Tomato - Bacterial Spot",
      "Tomato - Early Blight",
      "Potato - Healthy",
      "Maize - Leaf Blight",
      "Healthy Leaf"
  ]

  def predict_disease(image_path):
      return random.choice(DISEASES)
  ```

4. **Backend (Flask)**
- Built `app.py` to handle:
  - File uploads
  - Calling `predict_disease()`
  - Returning results to the frontend

5. **Frontend Template**
- Created `templates/index.html` with:
  - File upload form
  - Display for uploaded image
  - Display for prediction results

6. **Styling**
- Added `static/style.css` for basic page design.

7. **Testing**
- Ran the app locally (`python app.py`).
- Uploaded sample leaf images to confirm prediction display.

---

## Current Status
The current build uses a simulated AI model for demonstration purposes to ensure fast execution and compatibility without heavy dependencies. This design enables smooth operation even in low-resource environments.

A future version will include a fully trained deep learning model for accurate plant disease classification.

---

## Potential Impact
- **Early Detection:** Reduces crop losses by identifying diseases in their early stages.
- **Farmer Empowerment:** Enables self-diagnosis without specialist intervention.
- **Scalability:** Can be expanded to multiple crops, languages, and regions.

---

## Future Enhancements
- Integration of an actual AI model trained on the *PlantVillage Dataset*.
- Multi-language support for broader accessibility.
- Offline mode for rural areas with limited internet.
- Mobile application version for field use.

---

