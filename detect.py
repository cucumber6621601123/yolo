from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test/istockphoto-685889914-612x612.jpg")
results[0].show()