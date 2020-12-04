[![Travis CI Shield](https://travis-ci.com/mynameisvinn/FaceDetector.svg?branch=master)](https://travis-ci.com/github/mynameisvinn/FaceDetector)

# FaceOff
Detect static faces in video.

# Usage
Run `Tutorial.ipynb`. 

The core function is:
```python
from FaceOff import count_faces

folder = "dataset/608832786432738882426817735212"  # containing a series of images of cropped faces
count_faces(folder, label="not_static")  # returns 0 since we're looking at non-static faces
```

# Methodology 
## Finding contiguous snippets
FaceOff finds long, contiguous sequences of perceptually similar images. Static faces are defined by long (>10 frames), contiguous frames with low inter-frame variation.