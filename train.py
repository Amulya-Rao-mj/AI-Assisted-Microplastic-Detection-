from ultralytics import YOLO

model = YOLO(
    r"D:\MAJORPROJECT\runs\detect\results\microplastic_detection-2\weights\last.pt"
)

model.train(resume=True)