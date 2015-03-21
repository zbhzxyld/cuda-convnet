import numpy
import cPickle
from options import *

idx = 1

dict = cPickle.load(open('2.850', 'rb'))
layers = dict['model_state']['layers']

layername = 'conv1'
layerIdx=-1
for i in range(len(layers)):
    if(cmp(layers[i]['name'], layername)==0):
        layerIdx=i
        break

print(layerIdx)
print(layers[layerIdx]['name'])
