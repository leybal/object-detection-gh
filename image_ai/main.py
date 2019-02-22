from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "images/image1.jpg"),
                                             output_image_path=os.path.join(execution_path, "images/resRetinaImage1.jpg"))
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])


YOLO_detector = ObjectDetection()
YOLO_detector.setModelTypeAsYOLOv3()
YOLO_detector.setModelPath(os.path.join(execution_path , "models/yolo.h5"))
YOLO_detector.loadModel()

detections = YOLO_detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "images/image1.jpg"),
                                                  output_image_path=os.path.join(execution_path, "images/resYoloImage1.jpg"))
print("____YOLO____")
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
