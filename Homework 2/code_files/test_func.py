import sys
import torch
import json
from compute import test_data, test
from torch.utils.data import DataLoader
from bleu_eval import BLEU

model = torch.load('SavedModel/model0.h5', map_location=lambda storage, loc: storage)
filepath = '/scratch1/nsuresh/DL/testing_data/feat'
dataset = test_data(filepath)
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
training_json = 'MLDS_hw2_1_data/training_label.json'


