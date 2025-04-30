# Computer Vision Offside Detection System
This repository contains code to detect players in offside positions on soccer field from broadcast view. Offside detection in this repository is capable of being ran on a single frame or an entire video clip.

### Features
- **Detections**: Detect the ball, players and referees
- **Team Classification**: Identify teams by jersey color and direction of player by using average team posiiton
- **Direction of Play**: Determine the direction each team is going by averaging player positions from team classification
- **Classify Goalkeeper Team**: Add goalkeepers to proper team by using their x-position in conjunction with team classification
- **Offside Detection**: Transform positions to 2D plane and identify offside position by using player x-positions

## Tech Stack
- **Programming Language**: [Python](https://www.python.org/)
- **Computer Vision Model**: [YOLOv8](https://yolov8.com/)
- **Computer Vision Library**: [OpenCV](https://opencv.org/)
- **Machine Learning Library**: [PyTorch](https://pytorch.org/)

## Single Frame Output Example
![Computer Vision Offside Detection Single Frame Output](/example_output.png)

## Installation

### Prerequisites
- [Python3](https://www.python.org/downloads/)

### Steps to Install
1. Clone Repository
```bash
git clone https://github.com/JustinDosaj/cv-offside-detection.git
```

2. Install packages and dependencies
```bash
pip install
```

3. Add `/video` folder SOURCE_VIDEO_PATH = "../videos/soccer_video_offside_2.mp4" and add a `<filename>.mp4` containing a soccer broadcast clip

4. Navigate to `/notebooks/offisdes.ipynb` and change `SOURCE_VIDEO_PATH` to your mp4 file path

5. Perform step 3 for `/notebooks/offside-video.ipynb`

### Running Detection
1. Run single frame detection by navigating to `offsides.ipynb` and clicking run

2. Run detection for entire video clip by navigating to `offside-video.ipynb` and clicking run

## Known Limitations
1. System struggles to detect every player if multiple players are grouped too close together but likely need multiple camera angles to make any significant improvement to this
2. Team classification struggles on green & white jerseys due to the field color and lines. Reducing jersey color caluclation to only include upper half of body improved this slightly.
3. Failure to detect all players results in imbalanced teams, which in effects identifying direction of play (required to determine teams offsensive and defensive halves). Manually balancing teams by cutting players from the larger team improved this, but can still result in false positives. 
