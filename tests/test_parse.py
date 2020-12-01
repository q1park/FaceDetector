from unittest import TestCase
import os
import glob
from parse import main


class TestParse(TestCase):

    def test_main(self):
        _ = main("assets/short1.mp4")
        files = glob.glob('cropped/*.png')
        self.assertGreater(len(files), 0)