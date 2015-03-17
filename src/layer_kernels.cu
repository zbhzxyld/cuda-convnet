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

#include <assert.h>

#include <layer_kernels.cuh>


__global__ void kLabel2Weight(const float* labels, const uint len, const bool tri, const float w1, const float w2, float* weight) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

	int offset = y * len + x;
    if(y < len && x < len) {
		if (!tri || x > y) {
			if(labels[x] == labels[y])
				weight[offset] = w1;
			else
				weight[offset] = w2;
		} else {
			weight[offset] = 0.0f;
		}
    }
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * numCases + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * (label == ty);
        v = __fdividef(v, y_l[tidx]);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * ((label == ty) - y_l[tidx]);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
    
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    cutilCheckMsg("computeEltwiseMaxGrad: Kernel execution failed");
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    cutilCheckMsg("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
    delete &maxProbs;
}

void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregCostGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregCostGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    cutilCheckMsg("computeLogregGrad: Kernel execution failed");
}

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add) {
    int numCases = acts.getLeadingDim();
    int numOut = acts.getFollowingDim();

    assert(acts.isSameDims(actsGrad));
    assert(acts.isContiguous());
    assert(actsGrad.isContiguous());
    assert(target.isContiguous());
    assert(acts.isTrans());
    assert(actsGrad.isTrans());

    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(acts);
        kSoftmaxGrad<false><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    } else {
        kSoftmaxGrad<true><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    }
    cutilCheckMsg("computeSoftmaxGrad: Kernel execution failed");
}

void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregSoftmaxGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregSoftmaxGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    cutilCheckMsg("computeLogregSoftmaxGrad: Kernel execution failed");
}

float computeFisher2Cost(NVMatrix& labels, NVMatrix& data, float alpha) {
    int numCases = data.getNumCols(); 
    int numD = data.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!data.isTrans());
    assert(labels.isContiguous());
    assert(data.isContiguous());

	// labels to mask
	NVMatrix M(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float w1 = 2.0f / (n + diff);
	float w2 = 2.0f / (n - diff);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, w1, -w2, M.getDevData());

	// weights
	NVMatrix W(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, alpha, 1.0f - alpha, W.getDevData());
	float sum = W.sum();
	W.scale(1.0f / sum);

	// square: s1 = X^2
	NVMatrix s, s1;
	data.apply(NVMatrixOps::Square(), s);
	s.sum(0, s1);

	// cross: E = -2 * XT * X
	NVMatrix E;
	data.copy(s);
	s.transpose();
	s.rightMult(data, -2.0f, E);

	// distance: E = X^2 + X^2 - 2 * XT * X
	E.addVector(s1);
	s1.transpose();
	E.addVector(s1);

	// between class divergence: (sum(E .* M))^2
	M.eltwiseMult(E);
	float a = M.sum();

	// total divergence: sum(W .* (E - sum(E .* W))^2)
	E.eltwiseMult(W, M);
	float m = M.sum();
	E.addScalar(-m);
	E.apply(NVMatrixOps::Square());
	E.eltwiseMult(W);
	float b = E.sum();

	return -a * a / b;
}

void computeFisher2Grad(NVMatrix& labels, NVMatrix& data, float alpha, NVMatrix& target, bool add, float coeff)
{
	int numCases = data.getLeadingDim(); 
    int numD = data.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(data.isContiguous());
    assert(labels.isContiguous());
    assert(!data.isTrans());
	assert(!labels.isTrans());
    
    if (!add) {
        target.resize(data);
    }

	// labels to mask
	NVMatrix M(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float w1 = 2.0f / (n + diff);
	float w2 = 2.0f / (n - diff);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, w1, -w2, M.getDevData());

	// weights
	NVMatrix W(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, alpha, 1.0f - alpha, W.getDevData());
	float sum = W.sum();
	W.scale(1.0f / sum);
	
	NVMatrix tm, tew, tw;			// nx1
	NVMatrix xx, yy;				// dxn

	// square: s1 = X^2
	NVMatrix s1;
	data.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);

	// cross: E = -2 * XT * X
	NVMatrix E;
	data.copy(xx);
	xx.transpose();
	xx.rightMult(data, -2.0f, E);

	// distance: E = X^2 + X^2 - 2 * XT * X
	E.addVector(s1);
	s1.transpose();
	E.addVector(s1);

	// factors
	NVMatrix EM, EW;
	E.eltwiseMult(M, EM);
	float a = EM.sum();	// between class distance

	E.eltwiseMult(W, EW);
	float b = EW.sum();	// weighted mean

	E.addScalar(-b, EM);	// weighted covariance
	EM.apply(NVMatrixOps::Square());
	EM.eltwiseMult(W);
	a /= EM.sum();

	// 
	M.add(EW, -4.0f * a, 4.0f * a * a, EM);
	EM.add(W, -4.0f * a * a * b);

	// ew~, m~, w~
	M.sum(1, tm);
	tm.transpose();
	EW.sum(1, tew);
	tew.transpose();
	W.sum(1, tw);
	tw.transpose();
	tm.add(tew, 4.0f * a, -4.0f * a * a);
	tm.add(tw, 4.0f * a * a * b);

	// G_X
	data.eltwiseMultByVector(tm, xx);
	EM.transpose();
	data.rightMult(EM, s1);
	EM.transpose();
	xx.add(s1);

	// ew^, m^, w^
	M.sum(0, tm);
	EW.sum(0, tew);
	W.sum(0, tw);
	tm.add(tew, 4.0f * a, -4.0f * a * a);
	tm.add(tw, 4.0f * a * a * b);

	// G1_Y
	data.eltwiseMultByVector(tm, yy);
	data.rightMult(EM, s1);
	yy.add(s1);

	xx.add(yy);
	xx.scale(coeff);

	if (!add) {
		xx.copy(target);
	} else {
		target.add(xx);
	}
}


float getOptimalThr(Matrix& data, Matrix& labels, int nCases) {
	float curr, m=data(0,0);
	int i, j, rate, maxRate=0;
	for(i=0; i<nCases; i++) {
		curr = data(0,i);
		rate = 0;
		for(j=0; j<nCases; j++) {
			if(data(0,j)<curr && labels(0,j)==1 || data(0,j)>curr && labels(0,j)==-1)
				rate++;
		}
		if(rate>maxRate) {
			maxRate = rate;
			m = curr;
		}
	}
	return m;
}

/**
* L2SNCost:
* Loss = 0.5 * (gallery - probe) ^ 2, if label = 1;
*        0.5 * [m - |gallery - probe|2] ^ 2, if label = -1.
*/
void computeL2SNCost(NVMatrix& idens, NVMatrix& data, NVMatrix& output, float m) {
    int numCases = data.getNumCols();
    int numD = data.getNumRows();
	//float m = 0;

    assert(idens.getNumElements() == numCases);
	assert(numCases%2==0);
    assert(!idens.isTrans());
    assert(!data.isTrans());
    assert(idens.isContiguous());
    assert(data.isContiguous());

	NVMatrix gallery(numD, numCases/2);
	NVMatrix probe(numD, numCases/2);
	NVMatrix labels(1, numCases/2);
	NVMatrix iden_1(1, numCases/2);
	NVMatrix iden_2(1, numCases/2);
	output.resize(labels);

	data.slice(0, numD, 0, numCases/2, gallery);
	data.slice(0, numD, numCases/2, numCases, probe);
	idens.slice(0, 1, 0, numCases/2, iden_1);
	idens.slice(0, 1, numCases/2, numCases, iden_2);
	iden_1.equals(iden_2, labels);
	labels.apply(NVMatrixOps::WeightedAddScalar(2.0, -1.0));

	// calculate L2-norm of all pair samples: dists
	NVMatrix s1(gallery, true), s2(probe, true), temp(gallery, true), dists(labels, true);
	/*s1.apply(NVMatrixOps::Square());
	s2.apply(NVMatrixOps::Square());
	NVMatrix& magG = s1.sum(0);
	NVMatrix& magP = s2.sum(0);
	magG.apply(NVMatrixOps::Abs());
	magP.apply(NVMatrixOps::Abs());
	magG.apply(NVMatrixOps::Sqrt());
	magP.apply(NVMatrixOps::Sqrt());
	magG.addScalar(0.000001);
	magP.addScalar(0.000001);
	gallery.copy(s1);
	probe.copy(s2);
	s1.eltwiseDivideByVector(magG);
	s2.eltwiseDivideByVector(magP);
	delete &magG;
	delete &magP;*/

	//printf("1 galleries:\n");
	//s1.print(160,1);
	//printf("1 probes:\n");
	//s2.print(160,1);
	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	//temp.print(160,5);
	temp.sum(0, dists);
	temp.apply(NVMatrixOps::Abs());
	//dists.print(1,5);
	dists.apply(NVMatrixOps::Sqrt());
	dists.apply(NVMatrixOps::Abs());
	dists.addScalar(0.00000001);
	//printf("sqrt dists:\n");
	//dists.print(1,128);

	// calculate the optimal threshold: m
	if(m<0)
	{
		Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
		dists.copyToHost(devData);
		labels.copyToHost(devLabels);
		m = getOptimalThr(devData, devLabels, numCases/2);
	}
	//printf("%.3f ",m);
	//m = 3;

	// loss of positive samples: posLoss
	NVMatrix posMask(labels, true), posLoss(labels, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(0.5, 0.5), posMask);
	//posMask.print(1,128); //debug
	dists.apply(NVMatrixOps::Square(), posLoss);
	posLoss.apply(NVMatrixOps::MultByScalar(0.5));
	posLoss.eltwiseMult(posMask);
	//printf("posLoss:\n");
	//posLoss.print(1,128);  //debug

	// loss of negative samples: negLoss
	NVMatrix negMask1(labels, true), negMask2(labels, true), negLoss(labels, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(-0.5, 0.5), negMask1);
	//negMask1.print(1,128);  //debug
	dists.apply(NVMatrixOps::SmallerThanScalar(m), negMask2);
	dists.apply(NVMatrixOps::WeightedAddScalar(-1.0, m), negLoss);
	negLoss.apply(NVMatrixOps::Square());
	negLoss.apply(NVMatrixOps::MultByScalar(0.5));
	negLoss.eltwiseMult(negMask1);
	negLoss.eltwiseMult(negMask2);
	//printf("negLoss:\n");
	//negLoss.print(1,128);  //debug

	// add posLoss and negLoss: output
	posLoss.add(negLoss, output);
	//output.apply(NVMatrixOps::MultByScalar(coeff));
	//if(output.sum()<1)
	//output.print(1,128);  //debug
}

/**
* L2SNCost:
* Loss = 0.5 * (gallery - probe) ^ 2, if label = 1;
*        0.5 * [m - |gallery - probe|2] ^ 2, if label = -1.
*
* If label = 1:
*    G_X = probe - gallery
*    G_Y = gallery - probe
* if label = -1:
*    G_X = [m - |gallery - probe|2] / |gallery - probe|2 * (gallery - probe) or 0
*    G_Y = [m - |gallery - probe|2] / |gallery - probe|2 * (probe - gallery) or 0
*/


void computeL2SNGrad(NVMatrix& idens, NVMatrix& data, NVMatrix& dataTarget, float m,
	bool add, float coeff) {

    int numCases = data.getLeadingDim();
    int numOut = data.getFollowingDim();
	//float m = 0;

    assert(idens.getNumElements() == numCases);
	assert(numCases%2==0);
    assert(data.isContiguous());
    assert(idens.isContiguous());
    assert(!data.isTrans());
	assert(!idens.isTrans());

    if (!add) {
        dataTarget.resize(data);
    }

	NVMatrix gallery(numOut, numCases/2);
	NVMatrix probe(numOut, numCases/2);
	NVMatrix galleryTarget(numOut, numCases/2);
	NVMatrix probeTarget(numOut, numCases/2);
	NVMatrix tempTarget(numOut, numCases);
	NVMatrix labels(1, numCases/2);
	NVMatrix iden_1(1, numCases/2);
	NVMatrix iden_2(1, numCases/2);

	data.slice(0, numOut, 0, numCases/2, gallery);
	data.slice(0, numOut, numCases/2, numCases, probe);
	idens.slice(0, 1, 0, numCases/2, iden_1);
	idens.slice(0, 1, numCases/2, numCases, iden_2);
	iden_1.equals(iden_2, labels);
	labels.apply(NVMatrixOps::WeightedAddScalar(2.0, -1.0));

	// calculate L2-norm of all pair samples: dists
	NVMatrix s1(gallery, true), s2(probe, true), temp(gallery, true), dists(labels, true);
	/*s1.apply(NVMatrixOps::Square());
	s2.apply(NVMatrixOps::Square());
	NVMatrix& magG = s1.sum(0);
	NVMatrix& magP = s2.sum(0);
	magG.apply(NVMatrixOps::Abs());
	magP.apply(NVMatrixOps::Abs());
	magG.apply(NVMatrixOps::Sqrt());
	magP.apply(NVMatrixOps::Sqrt());
	magG.addScalar(0.000001);
	magP.addScalar(0.000001);
	gallery.copy(s1);
	probe.copy(s2);
	s1.eltwiseDivideByVector(magG);
	s2.eltwiseDivideByVector(magP);*/

	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	temp.sum(0, dists);
	temp.apply(NVMatrixOps::Abs());
	dists.apply(NVMatrixOps::Sqrt());
	dists.apply(NVMatrixOps::Abs());
	dists.addScalar(0.00000001);

	// calculate the optimal threshold: m
	if(m<0)
	{
		Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
		dists.copyToHost(devData);
		labels.copyToHost(devLabels);
		m = getOptimalThr(devData, devLabels, numCases/2);
	}
	//m = 3;

	// G_X_p, G_Y_p
	NVMatrix posMask(labels, true), diff1(gallery, true), diff2(gallery, true), G_X_p(gallery, true), G_Y_p(probe, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(0.5, 0.5), posMask);
	s2.subtract(s1, diff2);
	s1.subtract(s2, diff1);
	diff2.eltwiseMultByVector(posMask, G_X_p);
	diff1.eltwiseMultByVector(posMask, G_Y_p);
	//G_Y_p.print(160,5);

	// G_X_n, G_Y_n
	NVMatrix negMask1(labels, true), negMask2(labels, true), G_X_n(gallery, true), G_Y_n(probe, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(-0.5, 0.5), negMask1);
	dists.apply(NVMatrixOps::SmallerThanScalar(m), negMask2);
	dists.apply(NVMatrixOps::Reciprocal());
	dists.apply(NVMatrixOps::MultByScalar(m));
	dists.apply(NVMatrixOps::AddScalar(-1.0));
	diff1.eltwiseMultByVector(dists, G_X_n);
	diff2.eltwiseMultByVector(dists, G_Y_n);
	G_X_n.eltwiseMultByVector(negMask1);
	G_X_n.eltwiseMultByVector(negMask2);
	G_Y_n.eltwiseMultByVector(negMask1);
	G_Y_n.eltwiseMultByVector(negMask2);
	//G_Y_n.print(160,5);

	// G_X = G_X_p + G_X_n, G_Y = G_Y_p + G_Y_n
	G_X_p.add(G_X_n, galleryTarget);
    G_Y_p.add(G_Y_n, probeTarget);
	/*galleryTarget.eltwiseMultByVector(magG);
	probeTarget.eltwiseMultByVector(magP);*/
	//printf("galleryTarget\n");
	//galleryTarget.print(160,1);
	//printf("probeTarget\n");
	//probeTarget.print(160,1);
	galleryTarget.copy(tempTarget, 0, -1, 0, -1, 0, 0);
	probeTarget.copy(tempTarget, 0, -1, 0, -1, 0, numCases/2);
	tempTarget.apply(NVMatrixOps::MultByScalar(coeff));
	//printf("tempTarget\n");
	//tempTarget.print(160,1);

	if (!add) {
		tempTarget.copy(dataTarget);
	} else {
		dataTarget.add(tempTarget);
	}

	/*delete &magG;
	delete &magP;*/
}





/*
 * L2reg cost: L = 0.5 * (ground - predict).^2
 */
void computeL2regCost(NVMatrix& ground, NVMatrix& data, NVMatrix& output)
{
    int numCases = data.getNumCols();
    int numD = data.getNumRows();

    assert(ground.getNumCols() == numCases);
    assert(ground.getNumRows() == numD);
    assert(!ground.isTrans());
    assert(!data.isTrans());
    assert(ground.isContiguous());
    assert(data.isContiguous());

    // calculate L2-norm of all samples
    NVMatrix s(data, true); 
    NVMatrix loss(1, numCases, true);

    data.applyBinary(NVMatrixBinaryOps::SquaredDiff(), ground, s);
    s.sum(0, loss);

    output.resize(loss);
    loss.copy(output);
}

/*
 * L2reg cost: L = 0.5 * (ground - data).^2
 * G = ground - data
 */
void computeL2regGrad(NVMatrix& ground, NVMatrix& data, NVMatrix& target, 
    bool add, float coeff)
{
    int numCases = data.getNumCols();
    int numD = data.getNumRows();

    assert(ground.getNumCols() == numCases);
    assert(ground.getNumRows() == numD);
    assert(!ground.isTrans());
    assert(!data.isTrans());
    assert(ground.isContiguous());
    assert(data.isContiguous());
    
    NVMatrix diff(data, true);
    ground.subtract(data, diff);

    diff.apply(NVMatrixOps::MultByScalar(coeff));

    if (!add) {
        diff.copy(target);
    } else {
        target.add(diff);
    }
}