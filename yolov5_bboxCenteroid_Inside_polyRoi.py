import cv2
import numpy as np
import yolov5

# Load the YOLOv5 model
model_path = ' yolov5s.pt'

# device = "cpu"  # for cpu
device = 0  #for gpu
yolov5 = yolov5.YOLOv5(model_path,device,load_on_init=True)

# Load the video
video = cv2.VideoCapture("test22.mp4")

# Define the polygonal ROI
roi_points =  np.array([[100, 100], [400, 700], [1000, 500], [700, 200]], np.int32)


# Process each frame of the video
while True:
  # Read the next frame
  success, frame = video.read()
  if not success:
    break

  #draw poly
  cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

  # Perform object detection on the frame
  results = yolov5.predict(frame, size = 640, augment=False)
  detections = results.pred[0]


  # Check whether the bounding box centroids are inside the ROI
  for detection in detections:    
    xmin    = detection[0]
    ymin    = detection[1]
    xmax    = detection[2]
    ymax    = detection[3]
    score   = detection[4]
    class_id= detection[5]
    centroid_x = int(xmin + xmax) // 2
    centroid_y =  int(ymin + ymax) // 2


    # Check if the center of the bounding box is inside the polygon ROI
    if cv2.pointPolygonTest(roi_points, (centroid_x, centroid_y), False) > 0:
      # Perform desired action, such as drawing a circle around the centroid
      cv2.circle(frame, (centroid_x, centroid_y), 5, (0,255,0), -1)
      color = (0, 0, 225)
    else:
      color = (255, 0, 0)

    #Threshold score
    if score >= 0.5:
      cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)


  # Display the frame
  cv2.imshow("Video", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture object
video.release()
cv2.destroyAllWindows()
