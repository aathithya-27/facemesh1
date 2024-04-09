import cv2
import dlib
import numpy as np

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/FaceMeshBasics/FaceMeshBasics/.venv/Scripts/shape_predictor_68_face_landmarks.dat")  # Replace with correct path

# Function to detect facial landmarks
def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    landmarks = []
    for rect in rects[:5]:  # Limit to maximum of 5 faces
        shape = predictor(gray, rect)
        landmark_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        landmarks.append((landmark_points, rect))
    return landmarks

# Function to check if the face is real or fake
def is_real_face(landmarks):
    # For simplicity, let's just check if the mouth is open for each face
    real_faces = []
    for landmark, rect in landmarks:
        mouth_open = landmark[67, 1] - landmark[63, 1] > 10  # Check vertical distance between mouth corners
        real_faces.append((mouth_open, rect))
    return real_faces

# Main function
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_landmarks(frame)
        for landmark, face_rect in landmarks:
            if landmark is not None:
                if is_real_face([(landmark, face_rect)]):
                    label = "Real"
                    color = (0, 255, 0)  # Green for real faces
                else:
                    label = "Fake"
                    color = (0, 0, 255)  # Red for fake faces

                # Draw rectangle around the face and display label
                x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Anti-spoofing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
