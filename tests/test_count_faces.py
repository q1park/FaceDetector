from unittest import TestCase
from FaceOff import count_static_faces


class TestCount(TestCase):

    def test_count_static_faces(self):
        folder = "dataset/547538174468711490516541559363"
        self.assertEqual(count_static_faces(folder), 0)

        folder = "dataset/608832786432738882426817735212"
        self.assertEqual(count_static_faces(folder), 0)