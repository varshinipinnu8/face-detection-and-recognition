import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# ----------------------------------------------------
# Load Models (Loaded only once)
# ----------------------------------------------------

detector = MTCNN()
embedder = FaceNet()

# ----------------------------------------------------
# Detect Faces
# ----------------------------------------------------

def detect_faces(image):
    """
    Detect all faces in an image.

    Parameters:
        image (numpy.ndarray): RGB image

    Returns:
        list: List of detected faces
    """
    return detector.detect_faces(image)


# ----------------------------------------------------
# Draw Bounding Boxes
# ----------------------------------------------------

def draw_faces(image, faces):
    """
    Draw bounding boxes around detected faces.
    """
    output = image.copy()

    for face in faces:
        x, y, w, h = face["box"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        confidence = face["confidence"]

        cv2.putText(
            output,
            f"{confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output


# ----------------------------------------------------
# Extract Face
# ----------------------------------------------------

def extract_face(image, required_size=(160, 160)):
    """
    Extract the first detected face from image.
    """

    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]["box"]

    x = abs(x)
    y = abs(y)

    face = image[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.resize(face, required_size)

    return face


# ----------------------------------------------------
# Generate Face Embedding
# ----------------------------------------------------

def get_embedding(face):
    """
    Generate FaceNet embedding.
    """

    if face is None:
        return None

    face = face.astype("float32")

    embedding = embedder.embeddings([face])[0]

    return embedding


# ----------------------------------------------------
# Cosine Similarity
# ----------------------------------------------------

def cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity.
    """

    similarity = 1 - cosine(emb1, emb2)

    return similarity


# ----------------------------------------------------
# Euclidean Distance
# ----------------------------------------------------

def euclidean_distance(emb1, emb2):

    return np.linalg.norm(emb1 - emb2)


# ----------------------------------------------------
# Verify Faces
# ----------------------------------------------------

def verify_faces(image1, image2, threshold=0.70):
    """
    Compare two face images.

    Returns:
        similarity
        distance
        prediction
    """

    face1 = extract_face(image1)
    face2 = extract_face(image2)

    if face1 is None or face2 is None:
        return None, None, "Face Not Detected"

    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)

    similarity = cosine_similarity(emb1, emb2)

    distance = euclidean_distance(emb1, emb2)

    prediction = "Same Person" if similarity >= threshold else "Different Person"

    return similarity, distance, prediction


# ----------------------------------------------------
# Process Detection
# ----------------------------------------------------

def process_detection(image):
    """
    Detect faces and draw bounding boxes.
    """

    faces = detect_faces(image)

    output = draw_faces(image, faces)

    return output, len(faces)