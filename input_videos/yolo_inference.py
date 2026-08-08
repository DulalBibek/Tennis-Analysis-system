from ultralytics import YOLO

model=YOLO('yolov8x')

result=model.predict('/Users/bibekdulal/Desktop/Tennis_analysis/input_videos/image.png', save=True)

print(result)

print("Boxes")

for box in result[0].boxes:
    print(box)




