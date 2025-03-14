from ultralytics import YOLO

model = YOLO("best_3.pt")

results = model("test/istockphoto-1371845520-612x612.jpg")
results[0].show()