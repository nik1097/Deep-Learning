import sys
import torch
import json
from compute import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

model = torch.load('SavedModel/model0.h5', map_location=lambda storage, loc: storage)
filepath = '/scratch1/nsuresh/DL/testing_data/feat'
dataset = test_data(filepath)
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

model = model.cuda()
ss = test(testing_loader, model, i2w)

with open('output.txt', 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))
