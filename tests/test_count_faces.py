from unittest import TestCase
from FaceOff import count_faces


class TestCount(TestCase):

    def test_count_faces(self):
    	"""counts for not-static images should equal zero.
    	"""
        folder = "dataset/547538174468711490516541559363"
        self.assertEqual(count_faces(folder, label="not_static"), 0)

        folder = "dataset/608832786432738882426817735212"
        self.assertEqual(count_faces(folder, label="not_static"), 0)