from ultralytics import YOLO

model = YOLO(
    r"D:\MAJORPROJECT\runs\detect\results\microplastic_detection-2\weights\best.pt"
)

results = model(
    r"D:\MAJORPROJECT\dataset\images\test"
)

for result in results:
    result.show()