# Traffic Sign Detection in Vietnam

This is a project I built to detect traffic signs in Vietnam in real-time. I used YOLOv11 for this task. The goal is to detect common traffic signs on the street from dashcam videos.

> **Note:** When starting this project, I used the code structure from this repository as a reference to learn how to organize my files: [https://github.com/abcdef54/Traffic-Sign-Detection.git](https://github.com/abcdef54/Traffic-Sign-Detection.git).

---

## 1. Setup & Installation

I recommend using a virtual environment (`venv`) to avoid messing up your main Python installation.

```bash
# 1. Clone the repository
git clone https://github.com/MinhHuy128/Sign_Detection_VN.git
cd Sign_Detection_VN

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# 4. Install required libraries
pip install -r requirements.txt
```

---

## 2. Weights & Dataset

### Weights
- You can download my pre-trained `best.pt` file from here: [Google Drive Link](https://drive.google.com/drive/folders/1qN2bqpTarob5gHwzFFDFynjGxfZ8bFYb?usp=sharing) This folder also includes the corresponding validation results for your reference."
- Please put the `best.pt` file in the `models/signs/` folder.
- If you just want to see how it works, here are some demo videos I recorded: [Demo Videos](https://drive.google.com/drive/folders/1MlM4ENgLl2FDMqf4ExqZ0WA-2RhHMEO7?usp=sharing)

### Dataset
- The dataset I used is hosted on Roboflow. You can download it here: [VNTS-Merge v2 Dataset](https://universe.roboflow.com/nl-gt2le/vnts-merge/dataset/2).
- Extract the zip file into the `datasets/vnts-merge-2/` folder.

---

## 3. How to test it

To run the detection on a video and save the result, just run:

```bash
# Note: Replace 'your_video.mp4' with the actual path to your video file!
python main.py --input your_video.mp4 --show --save
```
It will open a window and draw boxes around the traffic signs. The output will be saved in `runs/inference/`.

---

## 4. What I learned

Doing this project taught me a lot of practical skills:
- **YOLO Models:** I learned how object detection models work and how to train YOLO on a custom dataset.
- **Python & OpenCV:** I got comfortable reading and writing videos using OpenCV.
- **Problem Solving:** I faced many bugs like Out of Memory errors or videos playing too slow, and I learned how to fix them (like changing OpenCV codecs to `avc1` to fix video saving bugs on Windows).
- **Git & Structuring:** I learned how to organize a machine learning project into separate folders instead of putting everything in one big notebook.

---

## 5. Pros and Cons of my model

### Pros:
- It runs very fast. On my laptop, it handles video frames in real-time smoothly.
- It detects big and clear traffic signs very accurately.

### Cons:
- When traffic signs are too far away or the video is blurry, the model sometimes misses them.
- It struggles a bit in low light or night-time videos because my dataset didn't have enough dark images.

---

## 6. How I would improve it

If I had more time, I would try to:
1. Add more data (especially night time and rainy weather images) to make it more robust.
2. Use object tracking (like DeepSORT) so the bounding boxes don't flicker from frame to frame.
3. Learn how to convert the model to TensorRT so it can run even faster on weaker computers.
