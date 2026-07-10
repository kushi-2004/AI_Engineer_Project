import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
print("The webcam starts working... Press 'q' on the keyboard to close it.")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame, stream=True)
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("The AI Engineer - Live Recording", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
cap.release()
cv2.destroyAllwindows()