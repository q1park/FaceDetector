# Static Face Detector
Detect static faces in videos.

# Usage
## 1) Detect Faces (Static and Non-static) in Video
First, find frames containing faces. In those frames, crop and save those frames as `png` images.

Run `parse.py` and pass the filename of the video.
```bash
python parse.py --f assets/long1.mp4
```
`parse.py` will generate `png` images containing faces in the folder `/cropped`.

## 2) Identifying Static Faces
Run `Computing Frame Similarity.ipynb`.

In essence, it will find long, contiguous sequences of images that are similar. That's because static faces will be contiguous sequences of images with low inter-frame variation.

### Computing Similarity
Similarity between image frames is computed with perceptual hashes. Images that are similar to each other will have similiar hashes; images with variation (perhaps due to distortions or deformations due to facial movement) will have different hashes.