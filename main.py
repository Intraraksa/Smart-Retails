import cv2
from ultralytics import YOLO

with open('util/map.txt', 'r') as f:
    goods = f.read()

RATIO = 0.5

model = YOLO("weight/best.pt")
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, (640, 480))
while cap.isOpened():
    _,frame = cap.read()
    # h = int(frame.shape[0]*RATIO)
    # w = int(frame.shape[1]*RATIO)
    # print(frame.shape)
    # frame_resized = cv2.resize(frame, (w,h))
    results = model.predict(source=frame, show=False, conf=0.75)
    for result in results:
        boxes = result.boxes
        if len(boxes) != 0:
            for box in boxes:
                x0,y0,x1,y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                classes = goods.split(',')[int(box.cls)]
                cv2.rectangle(frame,(int(x0),int(y0)),(int(x1),int(y1)),(200,120,165),3)
                cv2.putText(frame,f"{classes}",(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,1,(120,120,225),2)
                # print(boxes)
    cv2.imshow("Video", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()