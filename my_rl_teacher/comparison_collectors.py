import numpy as np
from collections import deque
from copy import deepcopy

class HumanComparisonCollector():
    def __init__(self):
        self.stack_num=1

        self.left_segment_dict={}
        self.left_segment_dict['observation']=deque(maxlen=self.stack_num)
        self.left_segment_dict['action']=deque(maxlen=self.stack_num)
        self.left_segment_dict['reward']=deque(maxlen=self.stack_num)

        self.right_segment_dict={}
        self.right_segment_dict['observation']=deque(maxlen=self.stack_num)
        self.right_segment_dict['action']=deque(maxlen=self.stack_num)
        self.right_segment_dict['reward']=deque(maxlen=self.stack_num)
        
        self.comparisons=[]

    def stack_segment(self,left,right):
        self.left_segment_dict['observation'].append(left['observation'])
        self.left_segment_dict['action'].append(left['action'])
        self.left_segment_dict['reward'].append(left['reward'])

        self.right_segment_dict['observation'].append(right['observation'])
        self.right_segment_dict['action'].append(right['action'])
        self.right_segment_dict['reward'].append(right['reward'])

    @property
    def stacked_segment_length(self):
        return len(self.left_segment_dict['observation'])

    @property
    def is_ready_add_segment_pair(self):
        return self.stacked_segment_length==self.stack_num

    def add_segment_pair(self, label):
        """Add a new unlabeled comparison from a segment pair"""

        assert self.stack_num==self.stacked_segment_length,\
        '''stacked segment length is lesser
        stack num: {0}
        stacked segment length: {1}'''\
        .format(self.stack_num,self.stacked_segment_length)
        
        comparison=None
        if label < 0:
            if self.stacked_comparisons_length <= 0:
                return
                
            comparison=np.random.choice(self.comparisons)

            if comparison['label'] == 0:
                comparison['right']=self.left_segment_dict if label == -1 else self.right_segment_dict
            if comparison['label'] == 1:
                comparison['left']=self.left_segment_dict if label == -1 else self.right_segment_dict

        else:
            comparison = {
                "left": deepcopy(self.left_segment_dict),
                "right": deepcopy(self.right_segment_dict),
                "label": label - 1
            }

        self.comparisons.append(comparison)

    @property
    def stacked_comparisons_length(self):
        return len(self.comparisons)

