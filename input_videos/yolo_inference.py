from ultralytics import YOLO

model=YOLO('yolov8x')

model.predict('/Users/bibekdulal/Desktop/Tennis_analysis/input_videos/image.png', save=True)

