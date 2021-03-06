# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy as n
import numpy.random as nr
from numpy.random import rand
import random
import itertools
import scipy.io
import h5py


WEB_STD_SIZE = 55

def Label2Class(labels, classNum):
    labels = n.require((labels), dtype=n.int32)
    
    classIndex = []
    for i in range(0, classNum):
        classIndex.append(n.array([]))        
        
    for i in range(0, len(labels)):
        classIndex[labels[i]] = n.append(classIndex[labels[i]], i)
        
    return classIndex

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 32
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        
        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
    
class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1

class DummyFaceDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, 48 * 48, num_classes = 2, num_cases = 512)
        
        self.num_colors = 1
        self.img_size = 48
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        #mask = (dic['labels'] == 0);
        #dic['labels'][mask] = -1;
        
        dic['vis'] = n.require(dic['data'].T, requirements='C')
        dic['nir'] = n.require(rand(self.num_cases, self.get_data_dims(1)).astype(n.single).T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        
        return epoch, batchnum, [dic['vis'], dic['nir'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx <= 1 else 1
        
# VIPER Database
class VIPERDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 48
        
        for d in self.data_dic:
            # remove mean
            d['a1'] = d['a'][:, :, 0] - self.batch_meta['a_m1']
            d['a2'] = d['a'][:, :, 1] - self.batch_meta['a_m2']
            d['a3'] = d['a'][:, :, 2] - self.batch_meta['a_m3']
            
            d['b1'] = d['b'][:, :, 0] - self.batch_meta['b_m1']
            d['b2'] = d['b'][:, :, 1] - self.batch_meta['b_m2']
            d['b3'] = d['b'][:, :, 2] - self.batch_meta['b_m3']
        
        nClass = len(self.batch_meta['label_names'])
        self.classIndex = range(0, nClass)
        self.classInfo = Label2Class(self.data_dic[0]['labels'][0], nClass)
        
        self.index = range(self.batch_meta['num_cases_per_batch'])
        
    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        
        # random shuffle
#        nr.shuffle(self.classInfo) 
#        self.index = []
#        for i in range(0, len(self.classInfo)):
#            self.index += self.classInfo[i].tolist()
        
        a1 = n.require(datadic['a1'][:, self.index], dtype=n.single, requirements='C')
        a2 = n.require(datadic['a2'][:, self.index], dtype=n.single, requirements='C')
        a3 = n.require(datadic['a3'][:, self.index], dtype=n.single, requirements='C')
        
        b1 = n.require(datadic['b1'][:, self.index], dtype=n.single, requirements='C')
        b2 = n.require(datadic['b2'][:, self.index], dtype=n.single, requirements='C')
        b3 = n.require(datadic['b3'][:, self.index], dtype=n.single, requirements='C')
        
        labels = n.require(datadic['labels'][:, self.index], dtype=n.single, requirements='C')
        
        return epoch, batchnum, [a1, b1, a2, b2, a3, b3, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 6 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)
      

# VIPER Database (single net)
class VIPERREPDataProvider(LabeledMemoryDataProvider):
    m = 0
    
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        #self.img_size = n.sqrt(n.array(self.batch_meta['num_vis']) / self.num_colors)
        self.img_size = 48
        
        # load mean
        mean = self.get_batch_meta(self.data_dir)
        #mean = self.get_batch_meta('C:/Users/Administrator/Documents/dyi/ftp/PR/CUHK/train')
        #VIPERREPDataProvider.m = mean['m'] 
        VIPERREPDataProvider.m = [mean['m1'], mean['m2'], mean['m3']]
            
        #if test:
        for d in self.data_dic:
            # merge two views of VIPER
            d['data'] = []
            for k in range(0, 3):
                temp = n.concatenate((d['a'][:, :, k], d['b'][:, :, k]), axis=1)
                temp = temp - VIPERREPDataProvider.m[k]
                d['data'] += [temp]
                
            d['labels'] = n.concatenate((d['labels'], d['labels']), axis=1)
            self.batch_meta['num_cases_per_batch'] = d['data'][0].shape[1]
            
#        else:
#            for d in self.data_dic:
#                # remove mean
#                d['data1'] = d['data1'] - VIPERREPDataProvider.m1
#                d['data2'] = d['data2'] - VIPERREPDataProvider.m2
#                d['data3'] = d['data3'] - VIPERREPDataProvider.m3

        nClass = int(max(self.data_dic[0]['labels'][0]) + 1)
        self.classIndex = range(0, nClass)
        self.classInfo = Label2Class(self.data_dic[0]['labels'][0], nClass)
        self.index = range(self.batch_meta['num_cases_per_batch'])

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        
        # random shuffle
        nr.shuffle(self.classInfo) 
        self.index = []
        for i in range(0, len(self.classInfo)):
            # random select 16 samples
#            sampleIndex = self.classInfo[i].tolist()
#            nr.shuffle(sampleIndex)
#            self.index += sampleIndex[0 : 16]
            self.index += self.classInfo[i].tolist()
        
        data = []
        for k in range(0, 3):
            data += [n.require(datadic['data'][k][:, self.index], dtype=n.single, requirements='C')]
        labels = n.require(datadic['labels'][:, self.index], dtype=n.single, requirements='C')  
        
#        data = []
#        for k in range(0, 3):
#            data += [n.require(datadic['data'][k], dtype=n.single, requirements='C')]
#        labels = n.require(n.zeros((1, data[0].shape[1])), dtype=n.single, requirements='C')  
        
        #return epoch, batchnum, [data[0], data[1], data[2], data[3], data[4], data[5], labels]
        return epoch, batchnum, [data[0], data[1], data[2], labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        #return self.img_size[idx]**2 * self.num_colors if idx < 6 else 1
        return self.img_size**2 * self.num_colors if idx < 3 else 1
    
    # Takes as input an array returned by get_next_batch
    # Re#turns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

# VIPER Database (for multi-task learning)
class VIPERMTDataProvider(LabeledMemoryDataProvider):
    m1 = 0
    m2 = 0
    m3 = 0
    
    CUHK_m1 = 0
    CUHK_m2 = 0
    CUHK_m3 = 0
    
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 48
        
        # load mean
        mean = self.get_batch_meta('C:/Users/Administrator/Documents/dyi/ftp/viper/dev')
        VIPERREPDataProvider.m1 = mean['m1']
        VIPERREPDataProvider.m2 = mean['m2']
        VIPERREPDataProvider.m3 = mean['m3']
        
        mean = self.get_batch_meta('C:/Users/Administrator/Documents/dyi/ftp/cuhk/train')
        VIPERREPDataProvider.CUHK_m1 = mean['m1']
        VIPERREPDataProvider.CUHK_m2 = mean['m2']
        VIPERREPDataProvider.CUHK_m3 = mean['m3']
           
        # load VIPER
        self.batch_meta['num_cases_per_batch'] = self.batch_meta['num_cases_per_batch'] * 2
        for d in self.data_dic:
            # merge two views of VIPER
            d['a-data1'] = n.concatenate((d['a'][:, :, 0], d['b'][:, :, 0]), axis=1)
            d['a-data2'] = n.concatenate((d['a'][:, :, 1], d['b'][:, :, 1]), axis=1)
            d['a-data3'] = n.concatenate((d['a'][:, :, 2], d['b'][:, :, 2]), axis=1)
            d['a-labels'] = n.concatenate((d['labels'], d['labels']), axis=1)
            
            # remove mean
            d['a-data1'] = d['a-data1'] - VIPERREPDataProvider.m1
            d['a-data2'] = d['a-data2'] - VIPERREPDataProvider.m2
            d['a-data3'] = d['a-data3'] - VIPERREPDataProvider.m3
        
        nClass = len(self.batch_meta['label_names'])
        self.classInfo = Label2Class(self.data_dic[0]['a-labels'][0], nClass)
        self.index = range(self.batch_meta['num_cases_per_batch'])
        
        # load CUHK
        self.data_dic1 = unpickle('C:/Users/Administrator/Documents/dyi/ftp/cuhk/train/data_batch_1')
        # remove mean
        self.data_dic1['data1'] = self.data_dic1['data1'] - VIPERREPDataProvider.CUHK_m1
        self.data_dic1['data2'] = self.data_dic1['data2'] - VIPERREPDataProvider.CUHK_m2
        self.data_dic1['data3'] = self.data_dic1['data3'] - VIPERREPDataProvider.CUHK_m3
        
        nClass = 1816
        self.classInfo1 = Label2Class(self.data_dic1['labels'][0], nClass)
        self.index1 = range(self.data_dic1['labels'].shape[1])
        
    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        
        # random shuffle (VIPER)
        #nr.shuffle(self.classInfo) 
        #self.index = []
        #for i in range(0, len(self.classInfo)):
        #    self.index += self.classInfo[i].tolist()
        
        a_data1 = n.require(datadic['a-data1'][:, self.index], dtype=n.single, requirements='C')
        a_data2 = n.require(datadic['a-data2'][:, self.index], dtype=n.single, requirements='C')
        a_data3 = n.require(datadic['a-data3'][:, self.index], dtype=n.single, requirements='C')
        a_labels = n.require(datadic['a-labels'][:, self.index], dtype=n.single, requirements='C')
        
        # random shuffle (CUHK)
        nr.shuffle(self.classInfo1) 
        self.index1 = []
        for i in range(0, len(self.classInfo)):
            self.index1 += self.classInfo1[i].tolist()
        self.index1 = self.index1[ : :2]
        
        b_data1 = n.require(self.data_dic1['data1'][:, self.index1], dtype=n.single, requirements='C')
        b_data2 = n.require(self.data_dic1['data2'][:, self.index1], dtype=n.single, requirements='C')
        b_data3 = n.require(self.data_dic1['data3'][:, self.index1], dtype=n.single, requirements='C')
        b_labels = n.require(self.data_dic1['labels'][:, self.index1], dtype=n.single, requirements='C')
        
        return epoch, batchnum, [a_data1, b_data1, a_data2, b_data2, a_data3, b_data3, a_labels, b_labels]
        
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 6 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

# Face representation learning
class FACEREPDataProvider(MemoryDataProvider):
   
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        MemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
        self.img_size = 40  
          
        for d in self.data_dic:
            #d['vis'] = n.require(-self.batch_meta['m'] + d['vis'], dtype=n.single, requirements='C')            
            d['data'] = n.require(-self.batch_meta['m'] + d['data'], dtype=n.single, requirements='C')
            #d['labels'] = n.zeros((1, d['data'].shape[1]))
            d['labels'] = n.require(d['labels'], dtype=n.single, requirements='C')
        
    def get_next_batch(self):
        epoch, batchnum, datadic = MemoryDataProvider.get_next_batch(self)
        
#        index = range(self.batch_meta['num_cases_per_batch'])
#        nr.shuffle(index)
#        
#        datadic['vis'] = n.require(datadic['vis'][:, index], dtype=n.single, requirements='C')
#        datadic['nir'] = n.require(datadic['nir'][:, index], dtype=n.single, requirements='C')
#        datadic['labels'] = n.require(datadic['labels'][:, index], dtype=n.single, requirements='C')
        
        return epoch, batchnum, [datadic['data'], datadic['labels']]
   

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)
        
    def get_num_classes(self):
        return len(self.batch_meta['label_names'])
        
# Face CCA learning
class FACECCADataProvider(MemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        MemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
        self.img_size = 48
        for d in self.data_dic:
            # random shuffle
            index = range(self.batch_meta['num_cases_per_batch'])
            #nr.shuffle(index)  
        
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            #d['X'] = n.require(d['X'][:, index], dtype=n.single, requirements='C')
            #d['Y'] = n.require(d['Y'][:, index], dtype=n.single, requirements='C')
            
            d['X'] = n.require(d['data'][:, index], dtype=n.single, requirements='C')
            d['Y'] = n.require(d['data'][:, index], dtype=n.single, requirements='C')

        
    def get_next_batch(self):
        epoch, batchnum, datadic = MemoryDataProvider.get_next_batch(self)
        
        return epoch, batchnum, [datadic['X'], datadic['Y']]
   

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

# Multi-modal Face learning
class FACEMMDataProvider(MemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        MemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
        self.img_size = 40

    def get_next_batch(self):
        epoch, batchnum, datadic = MemoryDataProvider.get_next_batch(self)
        
        miniBatchSize = 128
        
        # random select positive and negative sample pairs
#        index1 = random.sample(xrange(self.batch_meta['XPos'].shape[0]), self.batch_meta['num_cases_per_batch'] / 2)
#        index2 = random.sample(xrange(self.batch_meta['XNeg'].shape[0]), self.batch_meta['num_cases_per_batch'] / 2)
#        
#        XPos = n.reshape(self.batch_meta['XPos'][index1], (-1, miniBatchSize / 2))
#        YPos = n.reshape(self.batch_meta['YPos'][index1], (-1, miniBatchSize / 2))
#        XNeg = n.reshape(self.batch_meta['XNeg'][index2], (-1, miniBatchSize / 2))
#        YNeg = n.reshape(self.batch_meta['YNeg'][index2], (-1, miniBatchSize / 2))
#        
#        indexX = n.concatenate((XPos, XNeg), axis=1)
#        indexX = indexX.ravel()
#        
#        indexY = n.concatenate((YPos, YNeg), axis=1)
#        indexY = indexY.ravel()
#        
#        X = n.require(datadic['X'][:, indexX], dtype=n.single, requirements='C')
#        Xlabels =  n.require(datadic['Xlabels'][:, indexX], dtype=n.single, requirements='C')
#        
#        Y = n.require(datadic['Y'][:, indexY], dtype=n.single, requirements='C')
#        Ylabels =  n.require(datadic['Ylabels'][:, indexY], dtype=n.single, requirements='C')
        
        X = n.require(datadic['X'], dtype=n.single, requirements='C')
        Xlabels =  n.require(datadic['Xlabels'], dtype=n.single, requirements='C')
        
        return epoch, batchnum, [X, X, Xlabels, Xlabels]
   

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 2 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

# Face and Landmarks learning
class FACELMDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
        self.img_size = 100
            
    def get_next_batch(self):
        epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        
        # This converts the data matrix to single precision and makes sure that it is C-ordered
        #
        data = n.require(datadic['data'] - self.batch_meta['m'], dtype=n.single, requirements='C')
        #weights = n.require(datadic['occlusion'], dtype=n.single, requirements='C')
        labels = n.require(datadic['labels'], dtype=n.single, requirements='C')
       
        return epoch, batchnum, [data, labels]
   
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_size**2 * self.num_colors
        elif idx == 1:
            return 1
        else:
            return 49
    
    def get_num_classes(self):
        return len(self.batch_meta['label_names'])
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

def sample_wr(population, k): 
    "Chooses k random elements (with replacement) from a population" 
    n = len(population) 
    _random, _int = random.random, int  # speed hack 
    return [population[_int(_random() * n)] for i in itertools.repeat(None, k)] 
    
# Face verification
class FaceVerifyDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
        self.img_size = 100
        
    def get_next_batch(self):
        epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        
        # generate mask from label (remove self and mirrored correspondense)
        mask = (datadic['labels'] == datadic['labels'].T)
        mask = n.triu(mask, k=2)
        
        # get the index of positive sample pairs
        [YPos, XPos] = n.where(mask == 1)
        
        miniBatchSize = 128       
        # random select positive sample pairs
        num = int(self.batch_meta['num_cases_per_batch'] / miniBatchSize / 4) * miniBatchSize
        index = random.sample(xrange(XPos.shape[0]), num)
        temp = random.sample(xrange(datadic['data'].shape[1]), num * 2)
         
        XPos1 = n.reshape(XPos[index], (-1, miniBatchSize / 4))
        YPos1 = n.reshape(YPos[index], (-1, miniBatchSize / 4))
        
        XNeg = n.reshape(temp[0 : num], (-1, miniBatchSize / 4))
        YNeg = n.reshape(temp[num :  ], (-1, miniBatchSize / 4))
    
        index1 = n.concatenate((XPos1, XNeg, YPos1, YNeg), axis=1)
        index1 = index1.ravel()
        #
        data = n.require(datadic['data'][:, index1] - self.batch_meta['m'], dtype=n.single, requirements='C')
        labels =  n.require(datadic['labels'][:, index1], dtype=n.single, requirements='C')
        #weights = n.require(datadic['occlusion'][:, index1], dtype=n.single, requirements='C')
        
        return epoch, batchnum, [data, labels]
   
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_size**2 * self.num_colors
        elif idx == 1:
            return 1
        else:
            return 49
    
    def get_num_classes(self):
        return len(self.batch_meta['label_names'])
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

# age estimation
class AGEDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
        self.img_size = 100
        
        # load data
        self.data_dic = []
        for i in self.batch_range:
            self.data_dic += [self.get_batch(i)]
  
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require(d['data'] - self.batch_meta['m'], dtype=n.single, requirements='C')
            d['age'] = n.require(d['age'] / 30.0, dtype=n.single, requirements='C')
            d['gender'] = n.require(d['gender'], dtype=n.single, requirements='C')
            d['race'] = n.require(d['race'], dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]
        
        return epoch, batchnum, [self.data_dic[bidx]['data'], 
                                 self.data_dic[bidx]['age'], 
                                 self.data_dic[bidx]['gender'], 
                                 self.data_dic[bidx]['race']]
        
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2), dtype=n.single)

class WEBDataProvider_1(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 1
        self.img_size = WEB_STD_SIZE
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']] #dataIdx

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
        
        
class WEBDataProvider_3(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = WEB_STD_SIZE
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']] #dataIdx

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1         
        
        
class WEBDataProvider_4(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 4
        self.img_size = WEB_STD_SIZE
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']] #dataIdx

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1   

class XGW1SN4213DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 55
        
        #for d in self.data_dic:
            
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            #d['data'] = n.require(d['data'], dtype=n.single, requirements='C')
            #d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')-1

        
    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        
        # random shuffle
        #nr.shuffle(self.index) 
        #datadic = scipy.io.loadmat('G:\\byang\\40x40x2-SN\\data_train_%d.mat' % (batchnum))
        #datadic = scipy.io.loadmat('I:\\byang\\data\\DeepID2\\p2\\20+4\\data_train_%d.mat' % (batchnum))
        #datadic = scipy.io.loadmat('I:\\byang\\data\\DeepID2\\p2\\358+20\\data_batch_%d.mat' % (batchnum))
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        d={}
        d['data'] = n.require(-datadic['m']+datadic['imgs'], dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C') - 1
        return epoch, batchnum, [d['data'], d['labels']]
    

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4213              

        
# XGW - xgw-25crop - part2 - 55x55x3-SN
class XGW1SN2ndDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 55
        
        #for d in self.data_dic:
            
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            #d['data'] = n.require(d['data'], dtype=n.single, requirements='C')
            #d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')-1

        
    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        
        # random shuffle
        #nr.shuffle(self.index) 
        #datadic = scipy.io.loadmat('G:\\byang\\40x40x2-SN\\data_train_%d.mat' % (batchnum))
        #datadic = scipy.io.loadmat('I:\\byang\\data\\DeepID2\\p2\\20+4\\data_train_%d.mat' % (batchnum))
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        d={}
        d['data'] = n.require(-datadic['m']+datadic['imgs'], dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C') - 1
        return epoch, batchnum, [d['data'], d['labels']]
    

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4213


# XGW - xgw-25crop - part2 - 55x55x3-SN online
class XGW_DEEPID_ONLINE_DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        root_batch = h5py.File('%s/root_batch.mat'%(self.data_dir), 'r')
        self.num_colors = 3
        self.img_size = 55


        self.m = root_batch['m'][:,:].T
        self.imgs = root_batch['imgs'][:,:].T


        
    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        
        # random shuffle
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        index = n.concatenate(datadic['index'], axis=1)-1
        d = {}
        d['data'] = n.require(self.imgs[:,index] - self.m, dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')-1
        return epoch, batchnum, [d['data'], d['labels']]
    

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4131


# XGW - xgw-25crop - part2 - 55x55x3-SN online
class XGW_DEEPID_ATTR_ONLINE_DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        root_batch = h5py.File('%s/root_batch.mat'%(self.data_dir), 'r')
        root_batch_attr = scipy.io.loadmat('%s/root_batch_attr.mat'%(self.data_dir))
        self.num_colors = 3
        self.img_size = 55


        self.m = root_batch['m'][:,:].T
        self.imgs = root_batch['imgs'][:,:].T
        self.pose = root_batch_attr['pose'][0:2,:] #only need the pitch and yaw
        self.light = root_batch_attr['light'][:,:]
        self.expression = root_batch_attr['expression'][:,:]


    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        # random shuffle
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        index = n.concatenate(datadic['index'], axis=1)-1
        d = {}
        d['data'] = n.require(self.imgs[:,index] - self.m, dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')-1
        d['pose'] = n.require(self.pose[:,index], dtype=n.single, requirements='C')
        d['light'] = n.require(self.light[:,index], dtype=n.single, requirements='C')
        d['expression'] = n.require(self.expression[:,index], dtype=n.single, requirements='C')

        return epoch, batchnum, [d['data'], d['labels'],d['pose'],d['light'],d['expression']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix.
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4131




# XGW - xgw-25crop - part2 - 55x55x3-SN online
class XGW_DEEPID_ATTR_DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = 55


    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        d={}
        d['data'] = n.require(-datadic['m']+datadic['imgs'], dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C') - 1
        d['pose'] = n.require(datadic['pose'][0:2,:], dtype=n.single, requirements='C')
        d['light'] = n.require(datadic['light'], dtype=n.single, requirements='C')
        d['expression'] = n.require(datadic['expression'], dtype=n.single, requirements='C')
        return epoch, batchnum, [d['data'], d['labels'],d['pose'],d['light'],d['expression']]


    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix.
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4131



# XGW - xgw-25crop - part2 - 55x55x3-SN online
class XGW_ATTR_ONLINE_DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        root_batch = h5py.File('%s/root_batch.mat'%(self.data_dir), 'r')
        root_batch_attr = scipy.io.loadmat('%s/root_batch_attr.mat'%(self.data_dir))
        self.num_colors = 3
        self.img_size = 55


        self.m = root_batch['m'][:,:].T
        self.imgs = root_batch['imgs'][:,:].T
        self.pose = root_batch_attr['pose'][0:2,:] #only need the pitch and yaw
        self.light = root_batch_attr['light'][:,:]
        self.expression = root_batch_attr['expression'][:,:]


    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        # random shuffle
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        index = n.concatenate(datadic['index'], axis=1)-1
        d = {}
        d['data'] = n.require(self.imgs[:,index] - self.m, dtype=n.single, requirements='C')
        d['pose'] = n.require(self.pose[:,index], dtype=n.single, requirements='C')
        d['light'] = n.require(self.light[:,index], dtype=n.single, requirements='C')
        d['expression'] = n.require(self.expression[:,index], dtype=n.single, requirements='C')

        return epoch, batchnum, [d['data'],d['pose'],d['light'],d['expression']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix.
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4131



# XGW - xgw-25crop - part2 - 55x55x3-SN online
class XGW_DEEPID_10575_ONLINE_DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        root_batch = h5py.File('%s/root_batch.mat'%(self.data_dir), 'r')
        self.num_colors = 3
        self.img_size = 55


        self.m = root_batch['m'][:,:].T
        self.imgs = root_batch['imgs'][:,:].T



    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        # random shuffle
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        index = n.concatenate(datadic['index'], axis=1)-1
        d = {}
        d['data'] = n.require(self.imgs[:,index] - self.m, dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')-1
        return epoch, batchnum, [d['data'], d['labels']]


    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix.
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 10575


# XGW - xgw-25crop - part2 - 55x55x3-SN online
class XGW_DEEPID_ONLINE_65X65_4131_DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        root_batch = h5py.File('%s/root_batch_65x65.mat'%(self.data_dir), 'r')
        self.num_colors = 3
        self.img_size = 65


        self.m = root_batch['m'][:,:].T
        self.imgs = root_batch['imgs'][:,:].T



    def get_next_batch(self):
        #epoch, batchnum, datadic = DataProvider.get_next_batch(self)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        # random shuffle
        datadic = scipy.io.loadmat('%s/data_batch_%d.mat' % (self.data_dir, batchnum))
        index = n.concatenate(datadic['index'], axis=1)-1
        d = {}
        d['data'] = n.require(self.imgs[:,index] - self.m, dtype=n.single, requirements='C')
        d['labels'] = n.require(datadic['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')-1
        return epoch, batchnum, [d['data'], d['labels']]


    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix.
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx < 1 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require(data.T.reshape(data.shape[1], self.img_size, self.img_size), dtype=n.single)

    def get_num_classes(self):
        return 4131
