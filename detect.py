from ultralytics import YOLO

model = YOLO("best_3.pt")

results = model("test/istockphoto-1080293344-612x612.jpg")
results[0].show()