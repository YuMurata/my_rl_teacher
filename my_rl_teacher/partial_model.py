import yaml
import json
from pprint import pprint
from my_rl_teacher.predict_model import PredictModel

class PartialPredictModel:
    def __init__(self, structure_file:str):
        with open(structure_file,'r') as f:
            structure = json.load(f)
        pprint(data)
        print('partial: {0}'.format(len(data.items())))
        self.predict_models = {}

        for partial_name, index_list in data.items():
            self.predict_models[partial_name]=PredictModel()

if __name__ == "__main__":