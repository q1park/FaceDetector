import os
import glob
import re
import copy
import h5py
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
    
def make_dataset(h5_path, sample_duration):
    dataset = []
    with h5py.File(h5_path, 'r') as f:
        for uid, d in f.items():
            n_frames = d.shape[0]

            begin_t = 1
            end_t = n_frames
            sample = {
                'uid': uid,
                'vid_id':d.attrs['vid_id'], 
                'face_id':d.attrs['face_id'],
                'label':d.attrs['label'],
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
            }

            step = sample_duration
            for i in range(1, (n_frames - sample_duration + 1), step):
                sample_i = copy.deepcopy(sample)
                sample_i['frame_indices'] = list(range(i, i + sample_duration))
                sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
                dataset.append(sample_i)
    return dataset
    
def sort_video_urls(video_dir):
    re_meta = re.compile(r'(?<=\/)[0-9]+\.[0-9]+(?=\.)')

    df = pd.DataFrame([
            tuple([int(x) for x in re_meta.search(x).group().split('.')]+[x])
            for x in glob.glob(video_dir+'/*')
        ], columns=['frame', 'face', 'url']
    )
    
    url_lists = {
        face:df[df.face==face].sort_values(by='frame').url.tolist()
        for face in sorted(df.face.unique())
    }
    return url_lists

def make_video_array(url_list):
    dims = (380, 380, 3)
    shapes = set()
    video = np.empty((len(url_list), *dims)).astype(np.uint8)
    for i,path in enumerate(url_list):
        frame = cv2.imread(path)
        shapes.add(frame.shape)
        frame = cv2.resize(frame, (dims[0], dims[1]))
        video[i] = frame
    return video, shapes

def convert_to_h5(data_dir):
    video_dirs = [x for x in glob.glob(os.path.join(data_dir, *'**')) if re.search(r'[0-9]{5,20}', x)]
    re_labels = re.compile(r'(?<=[^0-9])[0-9]+\/.{0,4}static$')
#     print(video_dirs)
    shapes_all = set()
    face_id = 0
    
    with h5py.File(os.path.join(data_dir, 'data_all.h5'), 'w') as f:
        for video_dir in tqdm(video_dirs):
            vid_id, label = re_labels.search(video_dir).group().split('/')
            urls = sort_video_urls(video_dir)

            for face, (video, shapes) in zip(urls.keys(), map(make_video_array, urls.values())):
                shapes_all.update(shapes)
                d = f.create_dataset(str(face_id).zfill(5), data=video)
                d.attrs['vid_id'] = vid_id
                d.attrs['face_id'] = face
                d.attrs['label'] = 0 if label=='non_static' else 1
                
                face_id+=1
    print('unique shapes:', shapes)
    
def copy_dataset(h5, h5_og, name):
    h5.create_dataset(name, data=h5_og[name][:])
    h5[name].attrs.update({k:v for k,v in h5_og[name].attrs.items()})
#     print({k:v for k,v in h5_og[name].attrs.items()})
        
def train_test_split_face(data_dir, face_id):
    with h5py.File(os.path.join(data_dir, 'data_all.h5'), 'r') as f:
        f_train = h5py.File(os.path.join(data_dir, 'train_{}.h5'.format(face_id)), 'w')
        f_test = h5py.File(os.path.join(data_dir, 'test_{}.h5'.format(face_id)), 'w')
        
        alter = 1
        for uid,d in f.items():
            alter*=-1
            if d.attrs['face_id']!=int(face_id+alter):
                copy_dataset(f_train, f, uid)
            else:
                copy_dataset(f_test, f, uid)
        f_train.close()
        f_test.close()
    
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, device, h5_path, sample_duration, 
                 spatial_transform=None, temporal_transform=None):
        self.h5_path = h5_path
        self.data = make_dataset(self.h5_path, sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.device = device

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        clip = []
        with h5py.File(self.h5_path, 'r') as f:
            uid, frame_indices = self.data[index]['uid'], self.data[index]['frame_indices']
            
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
                
            for i in frame_indices:
                clip.append(Image.fromarray(f[uid][i]))

            if self.spatial_transform is not None:
                clip = [self.spatial_transform(img) for img in clip]


            video = torch.stack(clip, 0).permute(1, 0, 2, 3).to(self.device)
            label = torch.LongTensor([self.data[index]['label']]).to(self.device)
        return {'uid':uid, 'video':video, 'label':label}

    def __len__(self):
        return len(self.data)