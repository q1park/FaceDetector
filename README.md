# FaceOff
Detect static faces in videos.

# Usage
Run `Tutorial.ipynb`. 

The core function is:
```python
from FaceOff import count_faces

folder = "dataset/608832786432738882426817735212"
count_faces(folder, label="not_static")  # returns 0 since we're looking at non-static faces
```

# Methodology 
## Finding contiguous snippets
FaceOff finds long, contiguous sequences of similar images. Static faces will be contiguous sequences of images with low inter-frame variation.

## Computing Similarity
Similarity between image frames is computed with perceptual hashes. Images that are similar to each other will have similiar hashes; images with variation (perhaps due to distortions or deformations due to facial movement) will have different hashes.