import copy
import h5py
import torch
from PIL import Image

class H5Loader:
    def __init__(self, h5_path, sample_duration):
        self.h5 = h5py.File(h5_path, 'r')
        self.dataset = self.make_dataset(sample_duration)
        
    def make_dataset(self, sample_duration):
        dataset = []
        
        for uid, d in self.h5.items():
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

    def load_video(self, idx, temporal_transform=None):
        video = []
        uid, frame_indices = self.dataset[idx]['uid'], self.dataset[idx]['frame_indices']
        
        if temporal_transform is not None:
            frame_indices = temporal_transform(frame_indices)
            
        for i in frame_indices:
            video.append(Image.fromarray(self.h5[uid][i]))
        return video
    
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, sample_duration,
                 spatial_transform=None, temporal_transform=None):
        self.data = H5Loader(h5_path, sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
            
        clip = self.data.load_video(index, temporal_transform=self.temporal_transform)
        
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
            
        uid = self.data.dataset[index]['uid']
        video = torch.stack(clip, 0).permute(1, 0, 2, 3)
        label = self.data.dataset[index]['label']

        return {'uid':uid, 'video':video, 'label':label}

    def __len__(self):
        return len(self.data.dataset)