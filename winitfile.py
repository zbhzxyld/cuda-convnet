import numpy
import cPickle
from options import *

dict = cPickle.load(open('1.1700', 'rb'))
layers = dict['model_state']['layers']

def makew(name, idx, shape, params=None):
    layername = params[0]
    layerIdx=-1
    for i in range(len(layers)):
        if(cmp(layers[i]['name'], layername)==0):
            layerIdx=i
            break

    #if(cmp('conv1', layername)==0):
    #    aaa =  1
    return numpy.array(layers[layerIdx]['weights'][idx], dtype=numpy.single)
    
def makeb(name, shape, params=None):
    layername = params[0]
    layerIdx=-1
    for i in range(len(layers)):
        if(cmp(layers[i]['name'], layername)==0):
            layerIdx=i
            break

    return numpy.array(layers[layerIdx]['biases'], dtype=numpy.single)
    