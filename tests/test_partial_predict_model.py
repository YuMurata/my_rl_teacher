import unittest
from .context import partial_model
import tensorflow as tf

class TestPartialModel(unittest.TestCase):
    def setUp(self):
        summary_writer = tf.summary.FileWriter('summaries')
        self.score_partial_model = partial_model.PartialPredictModel('test.json',True, 'score', summary_writer)
        self.label_partial_model = partial_model.PartialPredictModel('test.json',False, 'label', summary_writer)



if __name__ == '__main__':
    unittest.main()