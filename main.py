# %%
from ultralytics import YOLO

# %%
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# %%
# Train the model
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# %%
model = YOLO('runs/detect/train/weights/best.pt')
# %%
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# %%
# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

# %%
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    # Display results to screen
    result.show()

    # Save results to disk
    result.save(filename=f'result_{i}.jpg') 
# %%
# Predict with the model
results = model('drone_0.jpg')  # predict on an image

# %%
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    # Display results to screen
    result.show()

    # Save results to disk
    result.save(filename=f'result_{i}.jpg') 
# %%
