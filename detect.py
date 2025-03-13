from ultralytics import YOLO

model = YOLO("best_1.pt")

results = model("test/istockphoto-505755464-612x612.jpg")
results[0].show()