Zebra Crossing Detection with YOLOv8 and OpenCV

Object detection using the YOLOv8 object detector for Zebra Crossing detection in both images and video streams using Deep Learning, OpenCV, and Python.

Unlike general-purpose detectors trained on the COCO dataset, this project focuses specifically on detecting zebra crossings in road images and video feeds.

🚦 Features

Detect zebra crossings in static images.

Detect zebra crossings in real-time video streams (CCTV, dashcam, or live feed).

Uses the state-of-the-art YOLOv8 deep learning model.

Fast and accurate with support for GPU acceleration.

🛠 Installation

Install the required dependencies:

pip install ultralytics
pip install opencv-python
pip install numpy

📸 YOLOv8 Object Detection in Images

To run detection on a single image:

python detect.py --weights best.pt --source images/zebra_crossing.jpg

Example Screenshot

(Add your screenshot here)

🎥 YOLOv8 Object Detection in Videos

To run detection on a video file:

python detect.py --weights best.pt --source videos/street.mp4 --save-txt --save-conf --project output --name zebra_output


To run detection on a live webcam stream:

python detect.py --weights best.pt --source 0

Example Screenshot / GIF

(Add your video detection screenshots here)

📂 Project Structure
├── images/                  # Input test images
├── videos/                  # Input test videos
├── output/                  # Detected output images/videos
├── best.pt                  # Trained YOLOv8 weights for zebra crossing detection
├── detect.py                # Detection script
└── README.md                # Project documentation

⚡ Real-time Zebra Crossing Detection

To run real-time detection:

python detect.py --weights best.pt --source 0


This will open a live webcam feed and detect zebra crossings in real-time.

📉 Limitation

Zebra crossings with low visibility (worn-out paint, poor lighting, shadows, or occlusion) may not be detected.

Performance depends on training data quality and camera resolution.

Works best on urban road environments with clear lane markings.

✅ Future Improvements

Improve detection under night and rainy conditions.

Add pedestrian detection together with zebra crossing recognition.

Deploy on edge devices (Raspberry Pi, Jetson Nano) for real-world smart traffic systems.

[LINK TO GOOGLE COLAB](https://colab.research.google.com/drive/1zPLVu-k9UUCD00-U2-20IfInraEjZ7RB?usp=chrome_ntp#scrollTo=-qguh0lgDVrd)
