import numpy
import cPickle
from options import *

dict = cPickle.load(open('../VIPER-Cross/train/64-fc500-cosine3-joint-alpha=2-offset=0.5-asym=2/240.1', 'rb'))
layers = dict['model_state']['layers']

def makew(name, idx, shape, params=None):
    layerIdx = int(params[0])
    return numpy.array(layers[layerIdx]['weights'][idx], dtype=numpy.single)
    
def makeb(name, shape, params=None):
    layerIdx = int(params[0])
    return numpy.array(layers[layerIdx]['biases'], dtype=numpy.single)
    