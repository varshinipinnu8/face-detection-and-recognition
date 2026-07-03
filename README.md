# 😊 Face Detection & Recognition Using MTCNN and FaceNet

A Deep Learning-based Face Detection and Face Verification system that detects human faces in images and verifies whether two face images belong to the same person using FaceNet embeddings and cosine similarity.

---

## 🚀 Features

- Detect multiple faces in an image
- Draw bounding boxes with confidence scores
- Generate FaceNet embeddings
- Verify whether two faces belong to the same person
- Display similarity score and Euclidean distance
- User-friendly web interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- OpenCV
- MTCNN
- FaceNet
- NumPy
- SciPy
- Streamlit

---

## 📂 Project Structure

```
Face-Detection-and-Recognition/
│── app.py
│── utils.py
│── requirements.txt
│── .python-version
│── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone  https://github.com/varshinipinnu8/face-detection-and-recognition

cd face-detection-and-recognition
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📷 Face Detection

1. Open the **Face Detection** module.
2. Upload an image.
3. The application detects all faces and displays:
   - Bounding boxes
   - Confidence score
   - Number of faces detected

---

## 👤 Face Verification

1. Open the **Face Verification** module.
2. Upload two face images.
3. Click **Verify Faces**.
4. The application displays:
   - Cosine Similarity
   - Euclidean Distance
   - Prediction (Same Person / Different Person)

---

## 🧠 Model Workflow

1. Upload image(s)
2. Detect faces using **MTCNN**
3. Extract facial region
4. Generate embeddings using **FaceNet**
5. Compare embeddings using **Cosine Similarity**
6. Display verification result

---

## 📊 Output

### Face Detection

- Detected Faces
- Bounding Boxes
- Confidence Score

### Face Verification

- Similarity Score
- Euclidean Distance
- Same Person / Different Person Prediction

---


Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🌐 Demo

```bash
  https://face-detection-and-recognition-1.onrender.com
```


---

## 👩‍💻 Author

**Varshini Pinnu**

