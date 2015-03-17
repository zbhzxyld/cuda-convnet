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
import numpy.random as nr
import numpy as n
import random as r
from copy import *

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

class FACEDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.test = test
        self.num_colors = 1
        self.img_size = 55
        # self.curr_epoch = 0
        # self.curr_batchnum = 1
        # self.batch_idx = -1

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        self.data = self.get_batch(self.curr_batchnum)
        self.data['data'] = n.require((self.data['data'] - self.data_mean), dtype=n.single, requirements='C')
        # if self.test:
        #     self.data['recon'] = self.data['data']
        # else:
        #     self.data['recon'] = n.require((self.data['recon'] - self.data_mean), dtype=n.single, requirements='C')
        self.data['recon'] = n.require((self.data['recon'] - self.data_mean), dtype=n.single, requirements='C')
        self.data["labels"] = n.c_[n.require(self.data['labels'], dtype=n.single)]
        self.data['labels'] = n.require(self.data['labels'].reshape((1, self.data['data'].shape[1])), dtype=n.single, requirements='C')
        if self.test:
            # In test/validation sessions, we need to know whether a pair is matched or mis-matched
            self.data['match'] = n.array(self.data['match'])
            self.data['match'] = n.require(self.data['match'].reshape((1, self.data['data'].shape[1])), dtype=n.single, requirements='C')
        if self.test:
            return epoch, batchnum, [self.data['data'], self.data['recon'], self.data['labels'], self.data['match']]
        else:
            return epoch, batchnum, [self.data['data'], self.data['recon'], self.data['labels']]

    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx in [0, 1] else 1

class FACEDataMemoryProvider(CIFARDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CIFARDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.test = test
        self.num_colors = 1
        self.img_size = 55
        for d in self.data_dic:
            d['recon'] = n.require((d['recon'] - self.data_mean), dtype=n.single, requirements='C')
        if self.test:
            # In test/validation sessions, we need to know whether a pair is matched or mis-matched
            for d in self.data_dic:
                d['match'] = n.array(d['match'])
                d['match'] = n.require(d['match'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        if self.test:
            return epoch, batchnum, [datadic['data'], datadic['recon'], datadic['labels'], datadic['match']]
        else:
            return epoch, batchnum, [datadic['data'], datadic['recon'], datadic['labels']]

    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx in [0, 1] else 1

class LFWView2MemoryProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=True):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.test = test
        self.num_colors = 1
        self.img_size = 55
        self.excl_mean_list = self.batch_meta['data_mean']
        for i in xrange(len(self.excl_mean_list)):
            self.excl_mean_list[i] = self.excl_mean_list[i].reshape((self.img_size*self.img_size, 1))
        self.data_dic = []
        if test==True:
            for i in batch_range:
                self.data_dic += [unpickle(self.get_data_file_name(i))]
                self.data_dic[-1]["labels"] = n.c_[n.require(self.data_dic[-1]['labels'], dtype=n.single)]
            for d in self.data_dic:
                d["labels"] = n.c_[n.require(d['labels'])]
                d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')
                d['match'] = n.array(d['match'])
                d['match'] = n.require(d['match'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]   # from 0 to 9

        self.exl_mean = self.excl_mean_list[bidx]
        one_fold = self.data_dic[bidx]
        one_fold['data'] = n.require((one_fold['data'] - self.exl_mean), dtype=n.single, requirements='C')
        one_fold['recon'] = n.require((one_fold['recon'] - self.exl_mean), dtype=n.single, requirements='C')
        one_fold = [one_fold['data'], one_fold['recon'], one_fold['labels'], one_fold['match']]

        nine_fold_list = [self.data_dic[i] for i in xrange(len(self.data_dic)) if i != bidx]
        nine_fold = copy(nine_fold_list[0])
        for nf in xrange(1,9):
            nine_fold['data'] = n.concatenate((nine_fold['data'], nine_fold_list[nf]['data']), axis=1)
            nine_fold['recon'] = n.concatenate((nine_fold['recon'], nine_fold_list[nf]['recon']), axis=1)
            nine_fold['labels'] = n.concatenate((nine_fold['labels'], nine_fold_list[nf]['labels']), axis=1)
            nine_fold['match'] = n.concatenate((nine_fold['match'], nine_fold_list[nf]['match']), axis=1)
        nine_fold['data'] = n.require((nine_fold['data'] - self.exl_mean), dtype=n.single, requirements='C')
        nine_fold['recon'] = n.require((nine_fold['recon'] - self.exl_mean), dtype=n.single, requirements='C')
        nine_fold = [nine_fold['data'], nine_fold['recon'], nine_fold['labels'], nine_fold['match']]

        return epoch, batchnum, one_fold, nine_fold

    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx in [0, 1] else 1

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
