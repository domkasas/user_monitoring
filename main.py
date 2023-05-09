from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet
from locker import Locker
import datetime
#import detect
import subprocess
import cv2

locker = Locker()
facenet = FaceNet(
    detector=MPFaceDetection(),
    onnx_model_path="models/faceNet.onnx",
    anchors="faces",
    threshold=0.3,
    force_cpu=True,
)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("Ignoring empty camera frame.")
            continue

        frame, face_crops = facenet(frame, draw=True)

        cv2.imshow('Video', frame)

        locker.onFaceNetPipeline(face_crops)
        if locker.deviceLocked and not locker.frame_saved:
            # save last frame
            print("Saving last frame")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
            filename = f"unauthorised/captured/{current_time}.jpg"
            cv2.imwrite(filename, frame)
            locker.lockDevice()
            locker.frame_saved = True
            subprocess.run(['python3', 'detect.py', '--image', filename])
            
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting")
            cv2.destroyAllWindows()
            break
        
    cap.release()

