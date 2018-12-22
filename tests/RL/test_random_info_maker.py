import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import tests.path_printer
import unittest
from my_rl_teacher.random_info_maker import make_random_info, make_random_info_list

class TestRandomInfoMaker(unittest.TestCase):
    def test_make_random_info(self):
        make_random_info()

if __name__ == '__main__':
    unittest.main()