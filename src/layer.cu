/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <algorithm>
#include <vector>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>

using namespace std;

/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNet* convNet, PyObject* paramsDict, bool trans) : 
             _convNet(convNet),  _trans(trans) {
    _name = pyDictGetString(paramsDict, "name");
    _type = pyDictGetString(paramsDict, "type");
    
    _numGradProducersNext = 0;
    _foundGradConsumers = false;
    _gradConsumer = (bool)pyDictGetInt(paramsDict, "gradConsumer");			// if this layer need gradients
    _actsTarget = pyDictGetInt(paramsDict, "actsTarget");				// the place to hold the activies
    _actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");		// the place to hold the gradients
    _conserveMem = (bool)pyDictGetInt(paramsDict, "conserveMem");				// release memory frequently
    _outputs = _actsTarget < 0 ? new NVMatrix() : NULL;
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : NULL;

	_dropout = pyDictGetFloat(paramsDict, "dropout");
    _dropoutMask = new NVMatrix();
}

void Layer::fpropNext(PASS_TYPE passType) {
    for (int i = 0; i < _next.size(); i++) {
        _next[i]->fprop(passType);
    }
}

void Layer::truncBwdActs() {
    // Only truncate actsGrad if I own it
    if (_conserveMem && _actsGradTarget < 0) { 
        getActsGrad().truncate();
    }
    if (_conserveMem) {
        getActs().truncate();
    }
}

void Layer::fprop(PASS_TYPE passType) {
    _rcvdFInputs += 1;
    if (_rcvdFInputs == _prev.size()) {
        NVMatrixV v;
        for (int i = 0; i < _prev.size(); i++) {
            v.push_back(&_prev[i]->getActs());
        }
        fprop(v, passType);
    }
}

void Layer::fprop(NVMatrix& v, PASS_TYPE passType) {
    NVMatrixV vl;
    vl.push_back(&v);
    fprop(vl, passType);
}

void Layer::fprop(NVMatrixV& v, PASS_TYPE passType) {
    assert(v.size() == _prev.size());
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
    _rcvdFInputs = (int)_prev.size();
    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);
    
    // First do fprop on the input whose acts matrix I'm sharing, if any
    if (_actsTarget >= 0) {
        fpropActs(_actsTarget, 0, passType);
    }
    // Then add the rest of the inputs to that
    for (int i = 0; i < _prev.size(); i++) {
        if (i != _actsTarget) {
            fpropActs(i, _actsTarget >= 0 || i > 0, passType);
        }
    }
    fpropNext(passType);

	// drop out 
	if (passType != PASS_TEST && _dropout > 0.0) {
        _dropoutMask->resize(getActs().getNumRows(), getActs().getNumCols());
        _dropoutMask->randomizeUniform();
        _dropoutMask->biggerThanScalar(_dropout);
        getActs().eltwiseMult(*_dropoutMask);
    }
      
    if (passType == PASS_TEST && _dropout > 0.0) {
        getActs().scale(1.0f - _dropout);
    }
}

void Layer::bprop(PASS_TYPE passType) {
    if (_rcvdBInputs == _numGradProducersNext) {
        _rcvdBInputs++; // avoid doing bprop computation twice
        bprop(getActsGrad(), passType);
    }
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType) {
    v.transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);

	// drop out
	if (_dropout > 0.0f) {
      v.eltwiseMult(*_dropoutMask);
    }
    
    bpropCommon(v, passType);
    
    if (isGradProducer()) {
        // First propagate activity gradient to all layers whose activity
        // gradient matrix I'm definitely not sharing.
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && _actsGradTarget != i) {
                bpropActs(v, i, _prev[i]->getRcvdBInputs() > 0 ? 1.0f : 0.0f, passType);
                _prev[i]->incRcvdBInputs();
            }
        }
        // Then propagate activity gradient to the layer whose activity gradient
        // matrix I'm sharing, if any.
        if (_actsGradTarget >= 0 && _prev[_actsGradTarget]->isGradConsumer()) {
            bpropActs(v, _actsGradTarget, _prev[_actsGradTarget]->getRcvdBInputs() > 0 ? 1.0f : 0.0f, passType);
            _prev[_actsGradTarget]->incRcvdBInputs();
        }
    }
    truncBwdActs();
    
    if (isGradProducer()) {
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer()) {
                _prev[i]->bprop(passType);
            }
        }
    }
}

void Layer::reset() {
    _rcvdFInputs = 0;
    _rcvdBInputs = 0;
}

string& Layer::getName() {
    return _name;
}

string& Layer::getType() {
    return _type;
}

int Layer::getRcvdFInputs() {
    return _rcvdFInputs;
}

int Layer::getRcvdBInputs() {
    return _rcvdBInputs;
}

int Layer::incRcvdBInputs() {
    return ++_rcvdBInputs;
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
    _numGradProducersNext += l->isGradProducer();
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

void Layer::postInit() {
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
    _actsGrad = _actsGradTarget < 0 ? new NVMatrix() : &_prev[_actsGradTarget]->getActsGrad();
}

// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
    if (!_foundGradConsumers) {
        for (int i = 0; i < _prev.size(); i++) {
            _gradConsumer |= _prev[i]->isGradConsumer();
        }
        _foundGradConsumers = true;
    }
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return true;
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    assert(_outputs != NULL);
    return *_outputs;
}

NVMatrix& Layer::getActsGrad() {
    assert(_actsGrad != NULL);
    return *_actsGrad;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNet* convNet, PyObject* paramsDict) 
    : Layer(convNet, paramsDict, true) {
    _neuron = &Neuron::makeNeuron(PyDict_GetItemString(paramsDict, "neuron"));
}

void NeuronLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->computeInputGrad(v, _prev[0]->getActsGrad(), scaleTargets > 0);
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->activate(*_inputs[0], getActs());
}

/* 
 * =======================
 * WeightLayer
 * =======================
 */
WeightLayer::WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans, bool useGrad) : 
    Layer(convNet, paramsDict, trans) {
    
    MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
    MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
    Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");
    
    floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
    float momB = pyDictGetFloat(paramsDict, "momB");
    floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
    float epsB = pyDictGetFloat(paramsDict, "epsB");
    floatv& wc = *pyDictGetFloatV(paramsDict, "wc");
    
    // Source layers for shared weights
    intv& weightSourceLayerIndices = *pyDictGetIntV(paramsDict, "weightSourceLayerIndices");
    // Weight matrix indices (inside the above source layers) for shared weights
    intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");
    
    for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
        int srcLayerIdx = weightSourceLayerIndices[i];
        int matrixIdx = weightSourceMatrixIndices[i];
        if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
            _weights.addWeights(*new Weights(_weights[matrixIdx], epsW[i]));
        } else if (srcLayerIdx >= 0) {
            WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
            Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
            _weights.addWeights(*new Weights(*srcWeights, epsW[i]));
        } else {
            _weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i], momW[i], useGrad));
        }
    }
    
    _biases = new Weights(hBiases, hBiasesInc, epsB, 0, momB, true);

    // Epsilons for finite-difference gradient checking operation
    _wStep = 0.001f;
    _bStep = 0.002f;
    
    delete &weightSourceLayerIndices;
    delete &weightSourceMatrixIndices;
    delete &hWeights;
    delete &hWeightsInc;
    delete &momW;
    delete &epsW;
    delete &wc;
}

void WeightLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    if (_biases->getEps() > 0) {
        bpropBiases(v, passType);
    }
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].getEps() > 0) {
            bpropWeights(v, i, passType);
            // Increment its number of updates
            _weights[i].incNumUpdates();
        }
    }
}

void WeightLayer::updateWeights() {
    _weights.update();
    _biases->update();
}

void WeightLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases->copyToGPU();
}

void WeightLayer::checkGradients() {
    for (int i = 0; i < _weights.getSize(); i++) {
        _convNet->checkGradient(_name + " weights[" + tostr(i) + "]", _wStep, _weights[i]);
    }
    _convNet->checkGradient(_name + " biases", _bStep, *_biases);
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights[idx];
}

Weights& WeightLayer::getBias() {
	return *_biases;
}

/* 
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNet* convNet, PyObject* paramsDict) : WeightLayer(convNet, paramsDict, true, false) {
    _wStep = 0.1f;
    _bStep = 0.01f;
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf("11\n");
    getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void FCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& weights_T = _weights[inpIdx].getW().getTranspose();
	//printf("12\n");
    _prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    delete &weights_T;
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumRows();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    _biases->getGrad().addSum(v, 0, 0, scaleBGrad);
}

void FCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumRows();

    NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    //printf("13\n");
    _weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);
    
    delete &prevActs_T;
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNet* convNet, PyObject* paramsDict, bool useGrad) 
    : WeightLayer(convNet, paramsDict, false, useGrad) {
    _padding = pyDictGetIntV(paramsDict, "padding");
    _stride = pyDictGetIntV(paramsDict, "stride");
    _filterSize = pyDictGetIntV(paramsDict, "filterSize");
    _channels = pyDictGetIntV(paramsDict, "channels");
    _imgSize = pyDictGetIntV(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _groups = pyDictGetIntV(paramsDict, "groups");
    _filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
    _randSparse = pyDictGetIntV(paramsDict, "randSparse");
    _overSample = pyDictGetIntV(paramsDict, "overSample");
    _filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
    _imgPixels = pyDictGetIntV(paramsDict, "imgPixels");
    
    _modulesX = pyDictGetInt(paramsDict, "modulesX");
    _modules = pyDictGetInt(paramsDict, "modules");

    // It's a vector on the heap to be consistent with all the others...
    _filterConns = new vector<FilterConns>();
    PyObject* pyFilterConns = PyDict_GetItemString(paramsDict, "filterConns");
    for (int i = 0; i < _randSparse->size(); i++) {
        FilterConns fc;
        if (_randSparse->at(i)) {
            fc.hFilterConns = getIntA(PyList_GET_ITEM(pyFilterConns, i));
        }
        _filterConns->push_back(fc);
    }
}

void LocalLayer::copyToGPU() {
    WeightLayer::copyToGPU();
    for  (int i = 0; i < _prev.size(); i++) {
        if (_randSparse->at(i)) { // Copy to GPU vector that describes sparse random connectivity
            cudaMalloc(&_filterConns->at(i).dFilterConns, sizeof(int) * _groups->at(i) * _filterChannels->at(i));
            cudaMemcpy(_filterConns->at(i).dFilterConns, _filterConns->at(i).hFilterConns,
                       sizeof(int) * _groups->at(i) * _filterChannels->at(i), cudaMemcpyHostToDevice);
            getLastCudaError ("cudaMemcpy: failed");
        }
    }
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict, true) {
    _partialSum = pyDictGetInt(paramsDict, "partialSum");
    _sharedBiases = (bool)pyDictGetInt(paramsDict, "sharedBiases");
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                             _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        convFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
    
    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(_biases->getW());
            getActs().reshape(_numFilters * _modules, getActs().getNumElements() / (_numFilters * _modules));
        } else {
            getActs().addVector(_biases->getW());
        }
    }
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
    }
}

void ConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();

    NVMatrix& tgt = _partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0; // ? 1 : 0;
    if (_randSparse->at(inpIdx)) {
        convWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt, _filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx), _modulesX, _modulesX,
                             _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    } else {
        convWeightActs(_prev[inpIdx]->getActs(), v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    }
    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }
}

void ConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        NVMatrix& tgt = _overSample->at(inpIdx) > 1 ? _actGradTmp : _prev[inpIdx]->getActsGrad();
        convImgActsSparse(v, *_weights[inpIdx], tgt, _filterConns->at(inpIdx).dFilterConns,
                          _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
                          _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
        if (_overSample->at(inpIdx) > 1) {
            _actGradTmp.reshape(_overSample->at(inpIdx), _actGradTmp.getNumElements() / _overSample->at(inpIdx));
            _actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
            _prev[inpIdx]->getActsGrad().reshape(_prev[inpIdx]->getActsGrad().getNumElements() / v.getNumCols(), v.getNumCols());
        }
    } else {
        convImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

void ConvLayer::truncBwdActs() {
    LocalLayer::truncBwdActs();
    if (_conserveMem) {
        _weightGradTmp.truncate();
        _actGradTmp.truncate();
    }
}
/* 
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict) : LocalLayer(convNet, paramsDict, false) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                              _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                        _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

    }  
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    
    float scaleInc = (passType != PASS_GC && _weights[inpIdx].getNumUpdates() == 0) * _weights[inpIdx].getMom(); // momentum
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases; // eps / numCases
    if (_randSparse->at(inpIdx)) {
        localWeightActsSparse(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx),
                              _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    } else {
        localWeightActs(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(), _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx),
                        _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    }
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localImgActsSparse(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _filterConns->at(inpIdx).dFilterConns,
                           _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                           _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx),  _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, true) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& input = *_inputs[0];
    NVMatrix& max = input.max(1);
    input.addVector(max, -1, getActs());
    getActs().apply(NVMatrixOps::Exp());
    NVMatrix& sum = getActs().sum(1);
    getActs().eltwiseDivideByVector(sum);
    
    delete &max;
    delete &sum;
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    bool doLogregGrad = _next.size() == 1 && _next[0]->getType() == "cost.logreg";
    if (doLogregGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
        computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
    } else {
        computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(), scaleTargets == 1);
    }
}

/* 
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _coeffs = pyDictGetFloatV(paramsDict, "coeffs");
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0) {
        _inputs[inpIdx]->scale(_coeffs->at(inpIdx), getActs());
    } else {
        getActs().add(*_inputs[inpIdx], _coeffs->at(inpIdx));
    }
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0 ) {
        v.scale(_coeffs->at(inpIdx), _prev[inpIdx]->getActsGrad());
    } else {
        assert(&_prev[inpIdx]->getActsGrad() != &v);
        _prev[inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
    }
}

/* 
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) { // First input, do nothing
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0], getActs());
    } else if (inpIdx > 1) {
        getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
    }
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}

/* 
 * =======================
 * EltwiseMulLayer
 * =======================
 */
EltwiseMulLayer::EltwiseMulLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void EltwiseMulLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf("EltMul: fp %f \n", scaleTargets);
    if (scaleTargets == 0) {
		_inputs[inpIdx]->copy(getActs());
    } else {
		getActs().eltwiseMult(*_inputs[inpIdx]);

		/*static int i = 0;
		char fileName[256];
		sprintf(fileName, "C:\\%d.txt", i++);
		Matrix temp1;
		getActs().copyToHost(temp1, true);
		FILE* fid = fopen(fileName, "wt");
		for (int i = 0; i < temp1.getNumElements(); i++) {
			fprintf(fid, "%f\n", temp1.getData()[i]);
		}
		fclose(fid);*/
    }
}

void EltwiseMulLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf("EltMul: bp %f \n", scaleTargets);
	if (scaleTargets == 0 ) {
		v.eltwiseMult(*_inputs[1 - inpIdx], _prev[inpIdx]->getActsGrad());
    } else {
        assert(&_prev[inpIdx]->getActsGrad() != &v);
		NVMatrix grad;
		v.eltwiseMult(*_inputs[1 - inpIdx], grad);
		_prev[inpIdx]->getActsGrad().add(grad, scaleTargets, 1.0f);
    }
}

/* 
 * =======================
 * ConcatLayer
 * =======================
 */
ConcatLayer::ConcatLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
	_dims = pyDictGetIntV(paramsDict, "dims");
	_d = 0;
	_offset.resize(_dims->size(), 0);
	for (int i = 0; i < _dims->size(); i++) {
		_d += _dims->at(i);
		if (i < _dims->size() - 1) 
			_offset[i + 1] = _offset[i] + _dims->at(i);
	}
}

void ConcatLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf("fp: %d\n", _offset[inpIdx]);
    if (scaleTargets == 0) {
		getActs().resize(_d, _inputs[inpIdx]->getLeadingDim());
    } 
	_inputs[inpIdx]->copy(getActs(), 0, -1, 0, -1, _offset[inpIdx], 0);
}

void ConcatLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//printf("bp: %d\n", _offset[inpIdx]);
	if (scaleTargets == 0 ) {
		_prev[inpIdx]->getActsGrad().resize(_dims->at(inpIdx), _inputs[inpIdx]->getLeadingDim());
		v.copy(_prev[inpIdx]->getActsGrad(), _offset[inpIdx], _offset[inpIdx] + _dims->at(inpIdx), 
			0, -1, 0, 0);
    } else {
		NVMatrix grad(_dims->at(inpIdx), _inputs[inpIdx]->getLeadingDim());
		v.copy(grad, _offset[inpIdx], _offset[inpIdx] + _dims->at(inpIdx), 
			0, -1, 0, 0);
		_prev[inpIdx]->getActsGrad().add(grad, scaleTargets, 1.0f);
    }
}

/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _dataIdx = pyDictGetInt(paramsDict, "dataIdx");
}

void DataLayer::fprop(PASS_TYPE passType) {
    throw string("No dava given!");
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
}

void DataLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
    _outputs = data[_dataIdx];
    fpropNext(passType);
}

bool DataLayer::isGradProducer() {
    return false;
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
	
	_pool = pyDictGetString(paramsDict, "pool");

	// max, agv, landmark
    _channels = pyDictGetInt(paramsDict, "channels");
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
	_imgSize = pyDictGetInt(paramsDict, "imgSize");

	// max, agv
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNet* convNet, PyObject* paramsDict) {
    string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNet, paramsDict);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNet, paramsDict);
	} else if(_pool == "landmark") {
		return *new LandmarkPoolLayer(convNet, paramsDict);
	}
    throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler());
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalAvgUndo(v, _prev[0]->getActsGrad(), _sizeX, _start, _stride, _outputsX, _imgSize, scaleTargets, 1);
}

/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(_prev[0]->getActs(), v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * LandmarkPoolLayer
 * =====================
 */
LandmarkPoolLayer::LandmarkPoolLayer(ConvNet* convNet, PyObject* paramsDict) : PoolLayer(convNet, paramsDict, false) {
}

void LandmarkPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLandmarkPool(*_inputs[0], *_inputs[1], getActs(), _channels, _sizeX, _outputsX, MaxPooler());
}

void LandmarkPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLandmarkMaxUndo(_prev[0]->getActs(), *_inputs[1], v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride, 0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, _start, _stride, scaleTargets, 1);
}

/* 
 * =====================
 * LandmarkLayer
 * =====================
 */
LandmarkLayer::LandmarkLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void LandmarkLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, 0, 6, 0, 1);
    convLandmark(*_inputs[0], *_inputs[1], getActs(), _channels, _imgSize, _outputsX, 0, 1);
	//printf("%dX%d\n", _inputs[1]->getNumRows(), _inputs[1]->getNumCols());
}

void LandmarkLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	//convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, 0, 6, scaleTargets, 1);
    convLandmarkUndo(v, *_inputs[1], _prev[0]->getActsGrad(), _channels, _imgSize, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
    convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void GaussianBlurLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& tgt1 = _prev[0]->getRcvdBInputs() > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    convGaussianBlur(v, _filter, tgt1, true, _channels, 0, 1);
    convGaussianBlur(tgt1, _filter, _prev[0]->getActsGrad(), false, _channels, scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}

/* 
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    _scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _center = (bool)pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");

    _scale = pyDictGetFloat(paramsDict, "scale");
    _pow = pyDictGetFloat(paramsDict, "pow");
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (_conserveMem) {
        _denoms.truncate();
    }
}

/* 
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _blocked = (bool)pyDictGetInt(paramsDict, "blocked");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _blocked);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, _blocked, scaleTargets, 1);
}


/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& images = *_inputs[0];
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, 1, _imgSize, AvgPooler());
    _meanDiffs.add(images, -1, 1);
    convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
    ResponseNormLayer::truncBwdActs();
    if (_conserveMem) {
        _meanDiffs.truncate();
    }
}

/* 
 * =====================
 * CrossMapResponseL2NormLayer
 * =====================
 */
CrossMapResponseL2NormLayer::CrossMapResponseL2NormLayer(ConvNet* convNet, PyObject* paramsDict) : ResponseNormLayer(convNet, paramsDict) {
}

void CrossMapResponseL2NormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseL2NormCrossMap(*_inputs[0], _denoms, getActs(), _channels);
}

void CrossMapResponseL2NormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseL2NormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(), _prev[0]->getActsGrad(), _channels, scaleTargets, 1);
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) 
    : Layer(convNet, paramsDict, trans) {
    _coeff = pyDictGetFloat(paramsDict, "coeff");
}

float CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop(PASS_TYPE passType) {
    if (_coeff != 0) {
        Layer::bprop(passType);
    }
}

bool CostLayer::isGradProducer() {
    return _coeff != 0;
}

doublev& CostLayer::getCost() {
    doublev& v = *new doublev();
    v.insert(v.begin(), _costv.begin(), _costv.end());
    return v;
}

CostLayer& CostLayer::makeCostLayer(ConvNet* convNet, string& type, PyObject* paramsDict) {
    if (type == "cost.logreg") {
        return *new LogregCostLayer(convNet, paramsDict);
    } else if (type == "cost.sum2") {
        return *new SumOfSquaresCostLayer(convNet, paramsDict);
	} else if (type == "cost.cosine") {
		return *new CosineCostLayer(convNet, paramsDict);
	} else if (type == "cost.cosine2") {
		return *new Cosine2CostLayer(convNet, paramsDict);
	} else if (type == "cost.cosine3") {
		return *new Cosine3CostLayer(convNet, paramsDict);
	} else if (type == "cost.cosine4") {
		return *new Cosine4CostLayer(convNet, paramsDict);
	} else if (type == "cost.cosine5") {
		return *new Cosine5CostLayer(convNet, paramsDict);
	} else if (type == "cost.agr") {
		return *new AGRCostLayer(convNet, paramsDict);
	} else if (type == "cost.fisher") {
		return *new FisherCostLayer(convNet, paramsDict);
	} else if (type == "cost.fisher2") {
		return *new Fisher2CostLayer(convNet, paramsDict);
	} else if (type == "cost.knife") {
		return *new KnifeCostLayer(convNet, paramsDict);
	} else if (type == "cost.knife2") {
		return *new Knife2CostLayer(convNet, paramsDict);
	} else if (type == "cost.dp") {
		return *new DPCostLayer(convNet, paramsDict);
	} else if (type == "cost.dp2") {
		return *new DP2CostLayer(convNet, paramsDict);
	} else if (type == "cost.attr") {
		return *new AttrCostLayer(convNet, paramsDict);
	} else if (type == "cost.l2") {
		return *new L2CostLayer(convNet, paramsDict);
	} else if (type == "cost.l2-sn") {
		return *new L2SNCostLayer(convNet, paramsDict);
	} else if (type == "cost.cosine-sn") {
		return *new CosineSNCostLayer(convNet, paramsDict);
	} else if (type == "cost.l3-sn") {
		return *new L3SNCostLayer(convNet, paramsDict);
	} else if (type == "cost.joint1") {
		return *new Joint1CostLayer(convNet, paramsDict);
	} else if (type == "cost.l2-reg") {
		return *new L2regCostLayer(convNet, paramsDict);
	}
    throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& trueLabelLogProbs = getActs(), correctProbs;
        computeLogregCost(labels, probs, trueLabelLogProbs, correctProbs);

		// for debug
		/*Matrix temp;
		probs.copyToHost(temp, true);
		FILE* fid = fopen("C:\\1.txt", "wt");
		for (int i = 0; i < temp.getNumElements(); i++) {
			fprintf(fid, "%f\n", temp.getData()[i]);
		}
		fclose(fid);*/

        _costv.clear();
        _costv.push_back(-trueLabelLogProbs.sum());
        _costv.push_back(numCases - correctProbs.sum());
    }
}

void LogregCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = _prev[0]->getActs();
    NVMatrix& probs = _prev[1]->getActs();
    NVMatrix& target = _prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax";
    if (doWork) {
        computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
    }
}

/* 
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _inputs[0]->apply(NVMatrixOps::Square(), getActs());
    _costv.clear();
    _costv.push_back(getActs().sum());
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -2 * _coeff);
}

/* 
 * =====================
 * CosineCostLayer
 * =====================
 */
CosineCostLayer::CosineCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void CosineCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& gallery = *_inputs[1];
		NVMatrix& probe = *_inputs[2];
        int numCases = labels.getNumElements();
        NVMatrix& output = getActs();
        computeCosineCost(labels, gallery, probe, output);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void CosineCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& gallery = _prev[1]->getActs();
		NVMatrix& probe = _prev[2]->getActs();
		NVMatrix& galleryTarget = _prev[1]->getActsGrad();
		NVMatrix& probeTarget = _prev[2]->getActsGrad();

		computeCosineGrad(labels, gallery, probe, galleryTarget, probeTarget, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * Cosine2CostLayer (matrix formulation, dual network)
 * =====================
 */
Cosine2CostLayer::Cosine2CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void Cosine2CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& gallery = *_inputs[1];
		NVMatrix& probe = *_inputs[2];
        int numCases = labels.getNumElements();
        float cost = computeCosine2Cost(labels, gallery, probe);

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void Cosine2CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& gallery = _prev[1]->getActs();
		NVMatrix& probe = _prev[2]->getActs();
		NVMatrix& galleryTarget = _prev[1]->getActsGrad();
		NVMatrix& probeTarget = _prev[2]->getActsGrad();

		computeCosine2Grad(labels, gallery, probe, galleryTarget, probeTarget, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * Cosine3CostLayer (deviance cost, single network)
 * =====================
 */
Cosine3CostLayer::Cosine3CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void Cosine3CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        float cost = computeCosine3Cost(labels, data);

		// for debug
		/*FILE* fid = fopen("c:\\cost.txt", "at");
		fprintf(fid, "%f\n", cost);
		fclose(fid);*/

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void Cosine3CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& target = _prev[1]->getActsGrad();

		computeCosine3Grad(labels, data, target, scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * Cosine4CostLayer (fisher cost, single network)
 * =====================
 */
Cosine4CostLayer::Cosine4CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_alpha = pyDictGetFloat(paramsDict, "alpha");
}

void Cosine4CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        float cost = computeCosine4Cost(labels, data, _alpha);

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void Cosine4CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& target = _prev[1]->getActsGrad();

		computeCosine4Grad(labels, data, _alpha, target, scaleTargets == 1, _coeff);
	}
}


/* 
 * =====================
 * Cosine5CostLayer (deviance cost, single network, multi-part score fusion)
 * =====================
 */
Cosine5CostLayer::Cosine5CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_alpha = pyDictGetFloat(paramsDict, "alpha");
	_beta = pyDictGetFloat(paramsDict, "beta");
	_gamma = pyDictGetFloat(paramsDict, "gamma");
}

void Cosine5CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data1 = *_inputs[1];
		NVMatrix& data2 = *_inputs[2];
		NVMatrix& data3 = *_inputs[3];
        int numCases = labels.getNumElements();
        float cost = computeCosine5Cost(labels, data1, data2, data3, 
			_alpha, _beta, _gamma);

		// for debug
		/*FILE* fid = fopen("c:\\cost.txt", "at");
		fprintf(fid, "%f\n", cost);
		fclose(fid);
*/
        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void Cosine5CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data1 = _prev[1]->getActs();
		NVMatrix& data2 = _prev[2]->getActs();
		NVMatrix& data3 = _prev[3]->getActs();
		NVMatrix& target1 = _prev[1]->getActsGrad();
		NVMatrix& target2 = _prev[2]->getActsGrad();
		NVMatrix& target3 = _prev[3]->getActsGrad();

		computeCosine5Grad(labels, data1, data2, data3, 
			_alpha, _beta, _gamma,
			target1, target2, target3, scaleTargets == 1, _coeff);
	}
}


/* 
 * =====================
 * FisherCostLayer (dual networks)
 * =====================
 */
FisherCostLayer::FisherCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void FisherCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& gallery = *_inputs[1];
		NVMatrix& probe = *_inputs[2];
        int numCases = labels.getNumElements();
        float cost = computeFisherCost(labels, gallery, probe);

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void FisherCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& gallery = _prev[1]->getActs();
		NVMatrix& probe = _prev[2]->getActs();
		NVMatrix& galleryTarget = _prev[1]->getActsGrad();
		NVMatrix& probeTarget = _prev[2]->getActsGrad();

		computeFisherGrad(labels, gallery, probe, galleryTarget, probeTarget, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * Fisher2CostLayer (single network)
 * =====================
 */
Fisher2CostLayer::Fisher2CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_alpha = pyDictGetFloat(paramsDict, "alpha");
}

void Fisher2CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        float cost = computeFisher2Cost(labels, data, _alpha);

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void Fisher2CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& target = _prev[1]->getActsGrad();

		computeFisher2Grad(labels, data, _alpha, target, 
			scaleTargets == 1, _coeff);
	}
}


/* 
 * =====================
 * KnifeCostLayer (Euclidian distance, single network)
 * =====================
 */
KnifeCostLayer::KnifeCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void KnifeCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        float cost = computeKnifeCost(labels, data);

		// for debug
		/*FILE* fid = fopen("c:\\cost.txt", "at");
		fprintf(fid, "%f\n", cost);
		fclose(fid);*/

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void KnifeCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& target = _prev[1]->getActsGrad();

		computeKnifeGrad(labels, data, target, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * Knife2CostLayer (Cosine similarity, single network)
 * =====================
 */
Knife2CostLayer::Knife2CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void Knife2CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        float cost = computeKnife2Cost(labels, data);

		// for debug
		/*FILE* fid = fopen("c:\\cost.txt", "at");
		fprintf(fid, "%f\n", cost);
		fclose(fid);*/

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void Knife2CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& target = _prev[1]->getActsGrad();

		computeKnife2Grad(labels, data, target, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * DPCostLayer (Dot product, dual networks)
 * =====================
 */
DPCostLayer::DPCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void DPCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& gallery = *_inputs[1];
		NVMatrix& probe = *_inputs[2];
        int numCases = labels.getNumElements();
        float cost = computeDPCost(labels, gallery, probe);

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void DPCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& gallery = _prev[1]->getActs();
		NVMatrix& probe = _prev[2]->getActs();
		NVMatrix& galleryTarget = _prev[1]->getActsGrad();
		NVMatrix& probeTarget = _prev[2]->getActsGrad();

		computeDPGrad(labels, gallery, probe, galleryTarget, probeTarget,
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * DP2CostLayer (Dot product, single network)
 * =====================
 */
DP2CostLayer::DP2CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void DP2CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        float cost = computeDP2Cost(labels, data);

        _costv.clear();
        _costv.push_back(cost * numCases);
    }
}

void DP2CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& target = _prev[1]->getActsGrad();

		computeDP2Grad(labels, data, target, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * MixedCostLayer for Age, Gender and Race
 * =====================
 */
AGRCostLayer::AGRCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void AGRCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& age = *_inputs[0];
		NVMatrix& gender = *_inputs[1];
		NVMatrix& race = *_inputs[2];
		NVMatrix& predict = *_inputs[3];
        int numCases = age.getNumElements();

        NVMatrix& output = getActs();
		NVMatrix ageLoss, genderLoss, raceLoss;
        computeAGRCost(age, gender, race, predict, ageLoss, genderLoss, raceLoss, output);
        _costv.clear();
        _costv.push_back(output.sum());
		_costv.push_back(ageLoss.sum());
		_costv.push_back(genderLoss.sum());
		_costv.push_back(raceLoss.sum());
    }
}

void AGRCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 3);

	NVMatrix& age = *_inputs[0];
	NVMatrix& gender = *_inputs[1];
	NVMatrix& race = *_inputs[2];
	NVMatrix& predict = *_inputs[3];

	NVMatrix& target = _prev[3]->getActsGrad();

	computeAGRGrad(age, gender, race, predict, target, 
		scaleTargets == 1, _coeff);
}

/* 
 * =====================
 * MixedCostLayer for 21 attributes
 * =====================
 */
AttrCostLayer::AttrCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void AttrCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& attr = *_inputs[0];
		NVMatrix& predict = *_inputs[1];
        int numCases = attr.getNumElements();

        NVMatrix& output = getActs();
        computeAttrCost(attr, predict, output);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void AttrCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 1);

	NVMatrix& attr = *_inputs[0];
	NVMatrix& predict = *_inputs[1];
	NVMatrix& target = _prev[1]->getActsGrad();

	computeAttrGrad(attr, predict, target, 
		scaleTargets == 1, _coeff);
}


/* 
 * =====================
 * L2CostLayer for dual network
 * =====================
 */
L2CostLayer::L2CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_m = pyDictGetFloat(paramsDict, "m");
}

void L2CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& gallery = *_inputs[1];
		NVMatrix& probe = *_inputs[2];
        int numCases = labels.getNumElements();
        NVMatrix& output = getActs();
        computeL2Cost(labels, gallery, probe, output, _m);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void L2CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& gallery = _prev[1]->getActs();
		NVMatrix& probe = _prev[2]->getActs();
		NVMatrix& galleryTarget = _prev[1]->getActsGrad();
		NVMatrix& probeTarget = _prev[2]->getActsGrad();

		computeL2Grad(labels, gallery, probe, galleryTarget, probeTarget, _m,
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * L2CostLayer for single network
 * =====================
 */
L2SNCostLayer::L2SNCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_m = pyDictGetFloat(paramsDict, "m");
}

void L2SNCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& output = getActs();
        computeL2SNCost(labels, data, output, _m);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void L2SNCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& dataTarget = _prev[1]->getActsGrad();

		computeL2SNGrad(labels, data, dataTarget, _m,
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * CosineCostLayer for single network
 * =====================
 */
CosineSNCostLayer::CosineSNCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void CosineSNCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& output = getActs();
        computeCosineSNCost(labels, data, output);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void CosineSNCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& dataTarget = _prev[1]->getActsGrad();

		computeCosineSNGrad(labels, data, dataTarget, 
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * L3CostLayer for single network
 * =====================
 */
L3SNCostLayer::L3SNCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_m = pyDictGetFloat(paramsDict, "m");
}

void L3SNCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
        int numCases = labels.getNumElements();
        NVMatrix& output = getActs();
        computeL3SNCost(labels, data, output, _m);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void L3SNCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& dataTarget = _prev[1]->getActsGrad();

		computeL3SNGrad(labels, data, dataTarget, _m,
			scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * Joint1CostLayer for single network
 * =====================
 */
Joint1CostLayer::Joint1CostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
	_m = pyDictGetFloat(paramsDict, "m");
	_lambda = pyDictGetFloat(paramsDict, "lambda");
}

void Joint1CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& data = *_inputs[1];
		NVMatrix& probs = *_inputs[2];
        int numCases = labels.getNumElements();
        NVMatrix& output = getActs();
        computeJoint1Cost(labels, data, probs, output, _m, _lambda);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void Joint1CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	if (inpIdx == 1) {
		NVMatrix& labels = _prev[0]->getActs();
		NVMatrix& data = _prev[1]->getActs();
		NVMatrix& dataTarget = _prev[1]->getActsGrad();
		NVMatrix& probs = _prev[2]->getActs();
		NVMatrix& probsTarget = _prev[2]->getActsGrad();

		computeJoint1Grad(labels, data, probs, dataTarget, probsTarget, _m, _lambda,
			scaleTargets == 1, _coeff);
	}
}





/* 
 * =====================
 * L2 CostLayer for vector regression
 * =====================
 */
L2regCostLayer::L2regCostLayer(ConvNet* convNet, PyObject* paramsDict) 
	: CostLayer(convNet, paramsDict, false) {
}

void L2regCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its three inputs together
    if (inpIdx == 0) {
        NVMatrix& ground = *_inputs[0];
		NVMatrix& predict = *_inputs[1];
        int numCases = ground.getNumElements();

        NVMatrix& output = getActs();
        computeL2regCost(ground, predict, output);
        _costv.clear();
        _costv.push_back(output.sum());
    }
}

void L2regCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 1);

	NVMatrix& ground = *_inputs[0];
	NVMatrix& predict = *_inputs[1];
	NVMatrix& target = _prev[1]->getActsGrad();

	computeL2regGrad(ground, predict, target, 
		scaleTargets == 1, _coeff);
}



/* 
 * =====================
 * ShiftLayer
 * =====================
 */
ShiftLayer::ShiftLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void ShiftLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convShift(*_inputs[0], _filter, getActs(), _channels);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void ShiftLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    //NVMatrix& tgt1 = _prev[0]->getRcvdBInputs() > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    //convShift(v, _filter, tgt1, _channels);
    //convShift(tgt1, _filter, _prev[0]->getActsGrad(), _channels);
}

void ShiftLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}



/* 
 * =====================
 * ShiftRandLayer
 * =====================
 */
ShiftRandLayer::ShiftRandLayer(ConvNet* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
	srand((int)time(NULL)); 
}

void ShiftRandLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convShiftRand(*_inputs[0], _filter, getActs(), _channels);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void ShiftRandLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    //NVMatrix& tgt1 = _prev[0]->getRcvdBInputs() > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    //convShift(v, _filter, tgt1, _channels);
    //convShift(tgt1, _filter, _prev[0]->getActsGrad(), _channels);
}

void ShiftRandLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}