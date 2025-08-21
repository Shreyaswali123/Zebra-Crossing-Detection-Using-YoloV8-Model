# 🦓 Zebra Crossing Detection using YOLOv8 Object Detection Method  

Real-time **Zebra Crossing Detection** using the **YOLOv8 object detector** in **Google Colab**, with support for **images, videos, and live streams**.  
Unlike generic detectors trained on the COCO dataset, this model is **specifically trained** for zebra crossing detection.  

---

## 🚦 Features  
- ✅ Detect zebra crossings in **static images**  
- ✅ Detect zebra crossings in **video streams** (CCTV, dashcam, traffic cameras)  
- ✅ **Real-time detection** with webcam or live feed  
- ✅ Trained & tested using **YOLOv8 on Google Colab**  
- ✅ Fast, accurate, and supports **GPU acceleration**  

---

## 🛠 Installation  

Install required dependencies in Colab or locally:  
```
pip install ultralytics
pip install opencv-python
pip install numpy
```
📂 Project Structure
```
├── custom_data/             # Training and validation dataset
│   ├── images/              # Images (train/validation split)
│   ├── labels/              # YOLO labels
│   └── classes.txt          # Class names file
├── data.yaml                # Dataset config file (auto-generated)
├── detect.py                # YOLO detection script
├── runs/                    # Training outputs (weights, logs, predictions)
├── best.pt                  # Trained YOLOv8 weights
└── README.md                # Project documentation
```

📸 Training in Google Colab
We trained the model using YOLOv8 (Nano variant) with a custom dataset.
Here’s the Colab training workflow:

# 📦 Unzip dataset into Colab
```
!unzip -q /content/data.zip -d /content/custom_data
```

# 📥 Download train/val split script
```
!wget -O /content/train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py
```
# 🔀 Split dataset (90% train, 10% validation)
```
!python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9
```
# 🚀 Install YOLOv8
```
!pip install ultralytics
```
🔧 Auto-generate data.yaml
```
import yaml, os
def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt not found at {path_to_classes_txt}')
        return
    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    data = {
        'path': '/content/data',
        'train': 'train/images',
        'val': 'validation/images',
        'nc': len(classes),
        'names': classes
    }
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Created config file at {path_to_data_yaml}')
create_data_yaml('/content/custom_data/classes.txt', '/content/data.yaml')
!cat /content/data.yaml
```

🏋️ Train YOLOv8
```
!yolo detect train data=/content/data.yaml model=yolov8n.pt epochs=40 imgsz=640
```
🎯 Inference (Prediction)
Detect Zebra Crossings in Validation Images
```
!yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True
```
Show Predictions in Colab
```
import glob
from IPython.display import Image, display
for image_path in glob.glob('/content/runs/detect/predict/*.jpg')[:15]:
    display(Image(filename=image_path, height=400))
    print('\n')
```
⚡ Real-Time Detection (Webcam / Video)
For video files:
```
python detect.py --weights best.pt --source videos/street.mp4 --save-txt --save-conf --project output --name zebra_output
```
For live webcam:
```
python detect.py --weights best.pt --source 0
```

📉 Limitations

❌ Low-visibility zebra crossings (worn paint, poor lighting, shadows) may not be detected.

❌ Performance depends on dataset quality and resolution.

✅ Works best on urban roads with clear markings.

✅ Future Improvements

🌙 Better detection in night & rainy conditions

🚶 Combine pedestrian detection with zebra crossing recognition

🔌 Deploy on edge devices (Raspberry Pi, Jetson Nano) for smart traffic systems

🔗 Resources

[YOLOv8 Documentation](https://docs.ultralytics.com/)

[Google Colab Notebook](https://colab.research.google.com/drive/1zPLVu-k9UUCD00-U2-20IfInraEjZ7RB?usp=chrome_ntp#scrollTo=-qguh0lgDVrd)
