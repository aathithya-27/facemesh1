import cv2
import mediapipe as mp
import time
import dlib
import numpy as np

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

        # Load face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("D:\FaceMeshBasics\FaceMeshBasics\.venv\Scripts\shape_predictor_68_face_landmarks.dat")  # Replace with correct path

    def detect_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        if len(rects) == 0:
            return None, None
        shape = self.predictor(gray, rects[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        return landmarks, rects[0]

    def is_real_face(self, landmarks):
        # Calculate additional features for anti-spoofing
        # For example, you could check if eyes are blinking, head movement, or facial expressions
        # Here's a simple example of checking the distance between eye landmarks
        left_eye_distance = np.linalg.norm(landmarks[42] - landmarks[45])
        right_eye_distance = np.linalg.norm(landmarks[36] - landmarks[39])

        # Determine if the eyes are open based on the distance between eye landmarks
        eyes_open = left_eye_distance > 5 and right_eye_distance > 5

        # Return True if the eyes are open, indicating a real face
        return eyes_open

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                               self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                landmarks = np.array(face)
                if self.is_real_face(landmarks):
                    label = "Real"
                    color = (0, 255, 0)  # Green for real faces
                else:
                    label = "Fake"
                    color = (0, 0, 255)  # Red for fake faces
                # Draw rectangle around the face and display label
                if draw:
                    face_rect = self.detect_landmarks(img)[1]
                    if face_rect:
                        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                 3, (255, 0, 0), 3)
        cv2.imshow("Anti-spoofing", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
