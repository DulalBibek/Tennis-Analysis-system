from ultralytics import YOLO

model=YOLO('/Users/bibekdulal/Desktop/Tennis_analysis/models/weights/last.pt')

result=model.predict('/Users/bibekdulal/Desktop/Tennis_analysis/input_videos/input_video.mp4',conf=0.2, save=True)

print(result)

print("Boxes")

for box in result[0].boxes:
    print(box)




