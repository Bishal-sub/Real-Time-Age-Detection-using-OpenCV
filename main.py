import cv2
import numpy as np


face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"


face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def detect_faces(net, frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            face_boxes.append([x1, y1, x2, y2])

    return face_boxes


def predict_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 MODEL_MEAN_VALUES, swapRB=False)

    age_net.setInput(blob)
    preds = age_net.forward()
    age = age_list[preds[0].argmax()]

    return age



cap = cv2.VideoCapture(0)   

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    boxes = detect_faces(face_net, frame)

    for (x1, y1, x2, y2) in boxes:
      
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        age = predict_age(face)

       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
        cv2.putText(frame, f"Age: {age}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

    cv2.imshow("Live Age Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()