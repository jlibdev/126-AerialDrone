import cv2
from ultralytics import YOLO

# Load your custom model
model = YOLO("pretrained/ArmyDroneDetection.pt")

# Open the video file
cap = cv2.VideoCapture(r"samplevideo\game.mp4")

if not cap.isOpened():
    print("Failed to open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

