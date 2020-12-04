from FaceOff import count_faces, get_actor_ids, group_frames_by_actors

from unittest import TestCase
from glob import glob
import os

class TestCount(TestCase):

    def test_count_faces(self):
        """counts for not-static images should equal zero.
        """
        label = "not_static"
        
        folder = "data/547538174468711490516541559363"
        fnames = sorted(glob(folder + '/' + label + '/*'))
        self.assertEqual(count_faces(fnames), 0)

        folder = "data/608832786432738882426817735212"
        fnames = sorted(glob(folder + '/' + label + '/*'))
        self.assertEqual(count_faces(fnames), 0)


    def test_static_faces(self):
        label = "static"
        folder = "data/547538174468711490516541559363"
        fnames = sorted(glob(folder + '/' + label + '/*'))
        self.assertGreater(count_faces(fnames), 0)


    def test_get_actor_ids(self):
        folder = "data/608832786432738882426817735212"
        label = "static"
        fnames = sorted(glob(folder + '/' + label + '/*'))
        self.assertEqual(len(get_actor_ids(fnames)), 11) 


    def test_group_frames_by_actors(self):
        folder = "data/608832786432738882426817735212"
        label = "static"
        fnames = sorted(glob(folder + '/' + label + '/*'))
        self.assertEqual(len(group_frames_by_actors(fnames).keys()), 11) 