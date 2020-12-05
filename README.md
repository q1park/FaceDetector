[![Travis CI Shield](https://travis-ci.com/mynameisvinn/FaceDetector.svg?branch=master)](https://travis-ci.com/github/mynameisvinn/FaceDetector)
[![codecov](https://codecov.io/gh/mynameisvinn/FaceDetector/branch/master/graph/badge.svg?token=PUSAA0P2CK)](https://codecov.io/gh/mynameisvinn/FaceDetector)

# FaceOff
Detect static faces in video.

# Usage
The core function is `count_faces(fnames)`. It accepts a `List[str]` (representing filenames of images) and returns an `int` (representing number of static faces).

```python
from FaceOff import count_faces

folder = "dataset/608832786432738882426817735212"  # images of cropped faces
label = "static"  # could be static, non-static or not-static

fnames = sorted(glob(folder + '/' + label + '/*'))  # list of image file names
count_faces(fnames)  # returns 0 since we're looking at non-static faces
```
An example can be found in `Tutorial.ipynb`. 

# Methodology 
## Finding contiguous snippets
FaceOff finds long, contiguous sequences of perceptually similar images. Static faces are defined by long (>10 frames), contiguous frames with low inter-frame variation.