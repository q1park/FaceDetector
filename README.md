# Static Face Detector
Detect static faces in videos.

## Basic Usage
### Extraction
```bash
python parse.py
```
This generates a series of images of faces in `/cropped`.
### Image Similarity
```bash
python similar.py
```
Still, static images in videos will have few variations from frame to frame, and therefore perceptual hashes of such images will be similar if not identical.