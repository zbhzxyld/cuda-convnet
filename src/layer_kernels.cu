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
#include <cula_lapack_device.h>

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
    
    getLastCudaError ("computeEltwiseMaxGrad: Kernel execution failed");
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
    getLastCudaError ("computeLogregCost: Kernel execution failed");
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

    getLastCudaError ("computeLogregGrad: Kernel execution failed");
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
    getLastCudaError ("computeSoftmaxGrad: Kernel execution failed");
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

    getLastCudaError ("computeLogregSoftmaxGrad: Kernel execution failed");
}


/*
 * hinge: L = max(0, 1 - cosine(gallery, probe) * labels)  -- cannot work
 * square: L = square(cosine(gallery, probe) - labels)
 * exp: L = exp(-cosine(gallery, probe) * labels)
 * binomial deviance: L = log(1 + exp(-2 * cosine(gallery, probe) * labels))
 * huber: L = Huber(cosine(gallery, probe) * labels - 1) 
 * 
 * labels:  (1, numCases)
 * gallery: (numD, numCases)
 * probe:   (numD, numCases) 
 * output:  (1, numCases)   (*out)
 */
void computeCosineCost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& output) {
    int numCases = gallery.getNumCols(); 
    int numD = gallery.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
    assert(labels.isContiguous());
    assert(gallery.isContiguous());
	assert(probe.isContiguous());

	output.resize(labels);

	// magnitude
	NVMatrix s1(gallery, true), s2(probe, true);
	s1.apply(NVMatrixOps::Square());
	s2.apply(NVMatrixOps::Square());
	NVMatrix& magXY = s1.sum(0);
	s2.sum(0, output);
	magXY.eltwiseMult(output);
	magXY.apply(NVMatrixOps::Sqrt());

	// cross correlation
	gallery.copy(s1);
	s1.eltwiseMult(probe);
	s1.sum(0, output);
    
	// normlization
	output.eltwiseDivide(magXY);
	delete &magXY;

	// hinge loss
	/*output.eltwiseMult(labels);
	output.apply(NVMatrixOps::MultByScalar(-1));
	output.apply(NVMatrixOps::AddScalar(1));
	output.maxWithScalar(0);*/

	// squared loss
	// output.applyBinary(NVMatrixBinaryOps::SquaredDiff(), labels);

	// exp loss
	/*output.eltwiseMult(labels);
	output.apply(NVMatrixOps::MultByScalar(-1));
	output.apply(NVMatrixOps::Exp());*/

	// binomial deviance
	output.eltwiseMult(labels);
	output.apply(NVMatrixOps::Deviance(2.0f));
}

/**
 * L = max(0, 1 - cosine(gallery, probe) * labels) -- cannot work
 * L = square(cosine(gallery, probe) - labels)
 * L = exp(-cosine(gallery, probe) * labels)
 * L = log(1 + exp(-2 * cosine(gallery, probe) * labels))
 * L = huber(cosine(gallery, probe) * labels) 
 * 
 * L(X, Y, Z) = ||X' * Y / (sqrt(X' * X) * sqrt(Y' * Y)) - Z||
 * G_X = 2 * L * G1_X
 * G_Y = 2 * L * G1_Y
 * G1_X = Y / magXY - X' * Y * X / (X' * X * magXY)
 * G1_Y = X / magXY - X' * Y * Y / (Y' * Y * magXY)
 * magXY = sqrt(X' * X) * sqrt(Y' * Y)
 */
void computeCosineGrad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff) {

    int numCases = gallery.getLeadingDim(); 
    int numOut = gallery.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(gallery.isContiguous());
    assert(probe.isContiguous());
    assert(labels.isContiguous());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
	assert(!labels.isTrans());

	// for debug
	// Matrix t1(500, 128), t2(500, 128);
    
    if (!add) {
        galleryTarget.resize(gallery);
        probeTarget.resize(probe);
    }

	// XTX, YTY 
	NVMatrix temp1(gallery, true), temp2(probe, true);
	temp1.apply(NVMatrixOps::Square());
	temp2.apply(NVMatrixOps::Square());
	NVMatrix& XTX = temp1.sum(0);
	NVMatrix& YTY = temp2.sum(0);

	// magnitude
	NVMatrix magXY(XTX, true);
	magXY.eltwiseMult(YTY);
	magXY.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	gallery.copy(temp1);
	temp1.eltwiseMult(probe);
	NVMatrix& XTY = temp1.sum(0);
    
	// L: normlization
	NVMatrix L(XTY, true);
	L.eltwiseDivide(magXY);
	
	// hinge loss
	/*labels.copy(L);
	L.apply(NVMatrixOps::MultByScalar(coeff));*/

	// square loss
	//L.subtract(labels);
	//L.apply(NVMatrixOps::MultByScalar(-2 * coeff));

	// exp loss
	/*L.eltwiseMult(labels);
	L.apply(NVMatrixOps::MultByScalar(-1));
	L.apply(NVMatrixOps::Exp());
	L.eltwiseMult(labels);
	L.apply(NVMatrixOps::MultByScalar(coeff));*/

	// binomial deviance
	L.eltwiseMult(labels);
	L.apply(NVMatrixOps::Logistic(2.0f));
	L.eltwiseMult(labels);
	L.apply(NVMatrixOps::MultByScalar(2 * coeff));

	// G1_X
	probe.copy(temp1);
	gallery.copy(temp2);
	temp2.eltwiseMultByVector(XTY);
	temp2.eltwiseDivideByVector(XTX);
	temp1.subtract(temp2);
	temp1.eltwiseDivideByVector(magXY);

	// G_X
	if (!add) {
		temp1.eltwiseMultByVector(L, galleryTarget);
	} else {
		temp1.eltwiseMultByVector(L);
		galleryTarget.add(temp1);
	}

	// G1_Y
	gallery.copy(temp1);
	probe.copy(temp2);
	temp2.eltwiseMultByVector(XTY);
	temp2.eltwiseDivideByVector(YTY);
	temp1.subtract(temp2);
	temp1.eltwiseDivideByVector(magXY);

	// G_Y
	if (!add) {
		temp1.eltwiseMultByVector(L, probeTarget);
	} else {
		temp1.eltwiseMultByVector(L);
		probeTarget.add(temp1);
	}

	delete &XTX;
	delete &YTY;
	delete &XTY;
}

float computeCosine2Cost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe) {
    int numCases = gallery.getNumCols(); 
    int numD = gallery.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
    assert(labels.isContiguous());
    assert(gallery.isContiguous());
	assert(probe.isContiguous());

	// labels to mask
	NVMatrix M(numCases, numCases);
	NVMatrix W1(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");

	float diff = M.sum();
	float n = numCases * numCases;
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f / (2.0f * n1), 1.0f / (2.0f * n2), W1.getDevData());
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -2.0f, M.getDevData());

	// magnitude
	NVMatrix xx, yy;
	gallery.apply(NVMatrixOps::Square(), xx);
	probe.apply(NVMatrixOps::Square(), yy);
	NVMatrix& s1 = xx.sum(0);
	NVMatrix& s2 = yy.sum(0);
	s1.apply(NVMatrixOps::Sqrt());
	s1.transpose();
	s2.apply(NVMatrixOps::Sqrt());

	// cross correlation
	NVMatrix xy;
	gallery.transpose();
	gallery.rightMult(probe, xy);
    
	// normlization
	xy.eltwiseDivideByVector(s1);
	xy.eltwiseDivideByVector(s2);
	delete &s1;
	delete &s2;

	// binomial deviance
	xy.addScalar(-0.5f);
	xy.eltwiseMult(M);
	xy.apply(NVMatrixOps::Deviance(2.0f));
	xy.eltwiseMult(W1);

	return xy.sum();
}

void computeCosine2Grad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff) {

    int numCases = gallery.getLeadingDim(); 
    int numOut = gallery.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(gallery.isContiguous());
    assert(probe.isContiguous());
    assert(labels.isContiguous());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
	assert(!labels.isTrans());

	// labels to mask
	NVMatrix M(numCases, numCases);
	NVMatrix W1(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");

	float diff = M.sum();
	float n = numCases * numCases;
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f / (2.0f * n1), 1.0f / (2.0f * n2), W1.getDevData());
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -2.0f, M.getDevData());

	// XTX, YTY 
	NVMatrix xx, yy;
	gallery.apply(NVMatrixOps::Square(), xx);
	probe.apply(NVMatrixOps::Square(), yy);
	NVMatrix& xtx = xx.sum(0);
	NVMatrix& yty = yy.sum(0);

	// magnitude
	xtx.apply(NVMatrixOps::Sqrt());
	xtx.transpose();
	yty.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	NVMatrix W;
	gallery.transpose();
	gallery.rightMult(probe, W);
	gallery.transpose();

	// normlization
	W.eltwiseDivideByVector(xtx);
	W.eltwiseDivideByVector(yty);

	// V1 and V2
	NVMatrix V1, V2;

	W.eltwiseDivideByVector(xtx, V1);
	V1.eltwiseDivideByVector(xtx);

	W.eltwiseDivideByVector(yty, V2);
	V2.eltwiseDivideByVector(yty);
    
	// W
	// binomial deviance
	W.addScalar(-0.5f);
	W.eltwiseMult(M);
	W.apply(NVMatrixOps::Logistic(2.0f));
	W.eltwiseMult(M);
	W.eltwiseMult(W1);
	W.apply(NVMatrixOps::MultByScalar(2 * coeff));

	// update U, V1, V2
	V1.eltwiseMult(W);
	V2.eltwiseMult(W);
	W.eltwiseDivideByVector(xtx);
	W.eltwiseDivideByVector(yty);
	
	// G_X
	V1.sum(1, xtx);
	xtx.transpose();
	gallery.eltwiseMultByVector(xtx, xx);
	W.transpose();
	probe.rightMult(W, yy);
	W.transpose();
	xx.add(yy, -1.0f, 1.0f);

	if (!add) {
		xx.copy(galleryTarget);
	} else {
		galleryTarget.add(xx);
	}

	// G_Y
	V2.sum(0, yty);
	probe.eltwiseMultByVector(yty, yy);
	gallery.rightMult(W, xx);
	yy.add(xx, -1.0f, 1.0f);
	if (!add) {
		yy.copy(probeTarget);
	} else {
		probeTarget.add(yy);
	}

	delete &xtx;
	delete &yty;
}

float computeCosine3Cost(NVMatrix& labels, NVMatrix& data) {
    int numCases = data.getNumCols(); 
    int numD = data.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!data.isTrans());
    assert(labels.isContiguous());
    assert(data.isContiguous());

	// labels to mask
	NVMatrix M(numCases, numCases);
	NVMatrix W(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");

	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f / (2 * n1), 1.0f / (2 * n2), W.getDevData());
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -2.0f, M.getDevData());

	// magnitude
	NVMatrix xx, s1;
	data.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);
	s1.apply(NVMatrixOps::Sqrt());
	
	// cross correlation
	NVMatrix xy;
	data.transpose(xx);
	xx.rightMult(data, xy);
    
	// normlization
	s1.transpose();
	xy.eltwiseDivideByVector(s1);
	s1.transpose();
	xy.eltwiseDivideByVector(s1);

	// binomial deviance
	xy.addScalar(-0.5f);
	xy.eltwiseMult(M);
	xy.apply(NVMatrixOps::Deviance(2.0f));
	xy.eltwiseMult(W);

	// exp loss
	/*xy.addScalar(-0.4f);
	xy.eltwiseMult(M);
	xy.apply(NVMatrixOps::MultByScalar(-5));
	xy.apply(NVMatrixOps::Exp());
	xy.eltwiseMult(W);*/

	return xy.sum();
}

void computeCosine3Grad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, 
	bool add, float coeff) {

    int numCases = data.getLeadingDim(); 
    int numOut = data.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(data.isContiguous());
    assert(labels.isContiguous());
    assert(!data.isTrans());
	assert(!labels.isTrans());

	// labels to mask
	NVMatrix M(numCases, numCases);
	NVMatrix W(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f / (2 * n1), 1.0f / (2 * n2), W.getDevData());
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -2.0f, M.getDevData());

	// XTX 
	NVMatrix xx, yy, xtx;
	data.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, xtx);

	// magnitude
	xtx.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	NVMatrix E;
	data.transpose(xx);
	xx.rightMult(data, E);

	// normlization
	xtx.transpose();		// i
	E.eltwiseDivideByVector(xtx);
	xtx.transpose();		// j
	E.eltwiseDivideByVector(xtx);

	// V1, V2
	NVMatrix V1, V2;
	
	xtx.transpose();		// i
	E.eltwiseDivideByVector(xtx, V1);
	V1.eltwiseDivideByVector(xtx);

	xtx.transpose();		// j
	E.eltwiseDivideByVector(xtx, V2);
	V2.eltwiseDivideByVector(xtx);
    
	// E
	// binomial deviance
	E.addScalar(-0.5f);
	E.eltwiseMult(M);
	E.apply(NVMatrixOps::Logistic(2.0f));
	E.eltwiseMult(M);
	E.eltwiseMult(W); 
	E.scale(2.0 * coeff);

	// exp loss
	/*E.addScalar(-0.4f);
	E.eltwiseMult(M);
	E.apply(NVMatrixOps::MultByScalar(-5));
	E.apply(NVMatrixOps::Exp());
	E.eltwiseMult(M);
	E.eltwiseMult(W);
	E.scale(5 * coeff);*/

	// update EU, EV
	V1.eltwiseMult(E);
	V2.eltwiseMult(E);

	E.eltwiseDivideByVector(xtx);
	xtx.transpose();		// i
	E.eltwiseDivideByVector(xtx);

	// G_X
	V1.sum(1, xtx);
	xtx.transpose();
	data.eltwiseMultByVector(xtx, xx);

	V2.sum(0, xtx);
	data.eltwiseMultByVector(xtx, yy);
	xx.add(yy);

	data.rightMult(E, yy);
	xx.add(yy, -1.0f, 1.0f);

	E.transpose();
	data.rightMult(E, yy);
	xx.add(yy);

	if (!add) {
		//printf("cosine3: 0\n");
		xx.copy(target);
	} else {
		//printf("cosine3: 1\n");
		target.add(xx);
	}
}

float computeCosine4Cost(NVMatrix& labels, NVMatrix& data, float alpha) {
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

	// positive weights
	NVMatrix W1(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, 0.0f, W1.getDevData());
	float sum = W1.sum();
	W1.scale(1.0f / sum);

	// negative weights
	NVMatrix W2(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 0.0f, 1.0f, W2.getDevData());
	sum = W2.sum();
	W2.scale(1.0f / sum);

	// magnitude
	NVMatrix xx;
	data.apply(NVMatrixOps::Square(), xx);
	NVMatrix& s1 = xx.sum(0);
	s1.apply(NVMatrixOps::Sqrt());

	// cross correlation
	NVMatrix xy;
	data.copy(xx);
	xx.transpose();
	xx.rightMult(data, xy);
    
	// normlization
	s1.transpose();	// i
	xy.eltwiseDivideByVector(s1);
	s1.transpose();	// j
	xy.eltwiseDivideByVector(s1);
	delete &s1;

	// between class
	M.eltwiseMult(xy);
	float a = M.sum();

	// weighted within class variance
	xy.eltwiseMult(W1, M);
	float m = M.sum();
	xy.addScalar(-m, M);
	M.apply(NVMatrixOps::Square());
	M.eltwiseMult(W1);
	float b = M.sum();

	xy.eltwiseMult(W2, M);
	m = M.sum();
	xy.addScalar(-m, M);
	M.apply(NVMatrixOps::Square());
	M.eltwiseMult(W2);
	float c = M.sum();

	return (alpha * b + (1.0f - alpha) * c) / (a * a);
}

void computeCosine4Grad(NVMatrix& labels, NVMatrix& data, float alpha, NVMatrix& target,  
	bool add, float coeff) {

    int numCases = data.getLeadingDim(); 
    int numOut = data.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(data.isContiguous());
    assert(labels.isContiguous());
    assert(!data.isTrans());
	assert(!labels.isTrans());

	NVMatrix MU, WU, WSU;
	NVMatrix mv1, wv1, wsv1;
	NVMatrix mv2, wv2, wsv2;

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

	// positive weights
	NVMatrix W1(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, 0, W1.getDevData());
	float sum = W1.sum();
	W1.scale(1.0f / sum);

	// negative weights
	NVMatrix W2(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 0.0f, 1.0f, W2.getDevData());
	sum = W2.sum();
	W2.scale(1.0f / sum);

	// XTX 
	NVMatrix xx, yy;
	data.apply(NVMatrixOps::Square(), xx);
	NVMatrix& xtx = xx.sum(0);

	// magnitude
	xtx.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	NVMatrix S;
	data.copy(xx);
	xx.transpose();
	xx.rightMult(data, S);

	// normlization
	xtx.transpose();	// i
	S.eltwiseDivideByVector(xtx);
	xtx.transpose();	// j
	S.eltwiseDivideByVector(xtx);

	/// positive term
	// factors
	S.eltwiseMult(W1, MU);
	float c = MU.sum();		// weighted mean

	S.eltwiseMult(M, MU);	
	float d = MU.sum();		// between class divergence
	float a = 1.0f / (d * d);

	S.addScalar(-c, MU);
	MU.apply(NVMatrixOps::Square());
	MU.eltwiseMult(W1);		// weighted variance
	float b = MU.sum() / (d * d * d);

	// U, V1, V2
	NVMatrix U(numCases, numCases), V1, V2;
	xtx.transpose();	// i
	S.eltwiseDivideByVector(xtx, V1);
	V1.eltwiseDivideByVector(xtx);
	xtx.transpose();	// j
	S.eltwiseDivideByVector(xtx, V2);
	V2.eltwiseDivideByVector(xtx);
	
	U.apply(NVMatrixOps::One());
	U.eltwiseDivideByVector(xtx);
	xtx.transpose();	// i
	U.eltwiseDivideByVector(xtx);

	// mv~, wv~, wsv~
	M.eltwiseMult(V1, MU);
	W1.eltwiseMult(V1, WU);
	WU.eltwiseMult(S, WSU);

	MU.sum(1, mv1);
	mv1.transpose();

	WU.sum(1, wv1);
	wv1.transpose();

	WSU.sum(1, wsv1);
	wsv1.transpose();

	// mv^, wv^, wsv^
	M.eltwiseMult(V2, MU);
	W1.eltwiseMult(V2, WU);
	WU.eltwiseMult(S, WSU);

	MU.sum(0, mv2);
	WU.sum(0, wv2);
	WSU.sum(0, wsv2);

	// add
	mv1.add(wsv1, 2.0f * b, -2.0f * a);
	mv1.add(wv1, 2.0f * a * c);

	mv2.add(wsv2, 2.0f * b, -2.0f * a);
	mv2.add(wv2, 2.0f * a * c);

	// MU, WU, WSU
	M.eltwiseMult(U, MU);
	W1.eltwiseMult(U, WU);
	WU.eltwiseMult(S, WSU);

	MU.add(WSU, 2.0f * b, -2.0f * a);
	MU.add(WU, 2.0f * a * c);

	MU.transpose(M);
	MU.add(M);
	
	// G_X
	data.eltwiseMultByVector(mv1, xx);
	xx.eltwiseMultByVector(mv2);

	data.rightMult(MU, yy);
	xx.add(yy, -1.0f, 1.0f);
	xx.scale(alpha * coeff);

	if (!add) {
		xx.copy(target);
	} else {
		target.add(xx);
	}

	/// negative term
	// factors
	S.eltwiseMult(W2, MU);
	c = MU.sum();		// weighted mean

	S.eltwiseMult(M, MU);	
	d = MU.sum();		// between class divergence
	a = 1.0f / (d * d);

	S.addScalar(-c, MU);
	MU.apply(NVMatrixOps::Square());
	MU.eltwiseMult(W2);		// weighted variance
	b = MU.sum() / (d * d * d);

	// U, V1, V2
	//NVMatrix U(numCases, numCases), V1, V2;
	xtx.transpose();	// i
	S.eltwiseDivideByVector(xtx, V1);
	V1.eltwiseDivideByVector(xtx);
	xtx.transpose();	// j
	S.eltwiseDivideByVector(xtx, V2);
	V2.eltwiseDivideByVector(xtx);
	
	U.apply(NVMatrixOps::One());
	U.eltwiseDivideByVector(xtx);
	xtx.transpose();	// i
	U.eltwiseDivideByVector(xtx);

	// mv~, wv~, wsv~
	M.eltwiseMult(V1, MU);
	W2.eltwiseMult(V1, WU);
	WU.eltwiseMult(S, WSU);

	MU.sum(1, mv1);
	mv1.transpose();

	WU.sum(1, wv1);
	wv1.transpose();

	WSU.sum(1, wsv1);
	wsv1.transpose();

	// mv^, wv^, wsv^
	M.eltwiseMult(V2, MU);
	W2.eltwiseMult(V2, WU);
	WU.eltwiseMult(S, WSU);

	MU.sum(0, mv2);
	WU.sum(0, wv2);
	WSU.sum(0, wsv2);

	// add
	mv1.add(wsv1, 2.0f * b, -2.0f * a);
	mv1.add(wv1, 2.0f * a * c);

	mv2.add(wsv2, 2.0f * b, -2.0f * a);
	mv2.add(wv2, 2.0f * a * c);

	// MU, WU, WSU
	M.eltwiseMult(U, MU);
	W2.eltwiseMult(U, WU);
	WU.eltwiseMult(S, WSU);

	MU.add(WSU, 2.0f * b, -2.0f * a);
	MU.add(WU, 2.0f * a * c);

	MU.transpose(M);
	MU.add(M);
	
	// G_X
	data.eltwiseMultByVector(mv1, xx);
	xx.eltwiseMultByVector(mv2);

	data.rightMult(MU, yy);
	xx.add(yy, -1.0f, 1.0f);
	xx.scale((1.0f - alpha) * coeff);

	target.add(xx);

	delete &xtx;
}

float computeCosine5Cost(NVMatrix& labels, NVMatrix& data1, NVMatrix& data2, NVMatrix& data3, 
	float alpha, float beta, float gamma)
{
	int numCases = data1.getNumCols(); 
    int numD = data1.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!data1.isTrans());
    assert(labels.isContiguous());
    assert(data1.isContiguous());

	// labels to mask
	NVMatrix M(numCases, numCases);
	NVMatrix W(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");

	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f / (2 * n1), 1.0f / (2 * n2), W.getDevData());

	//printf("1\n");
	///// part1
	// magnitude
	NVMatrix xx, s1;
	data1.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);
	s1.apply(NVMatrixOps::Sqrt());
	
	// cross correlation
	NVMatrix xy1;
	data1.transpose(xx);
	xx.rightMult(data1, xy1);
    
	// normlization
	s1.transpose();
	xy1.eltwiseDivideByVector(s1);
	s1.transpose();
	xy1.eltwiseDivideByVector(s1);

	//printf("2\n");
	///// part2
	// magnitude
	data2.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);
	s1.apply(NVMatrixOps::Sqrt());
	
	// cross correlation
	NVMatrix xy2;
	data2.transpose(xx);
	xx.rightMult(data2, xy2);
    
	// normlization
	s1.transpose();
	xy2.eltwiseDivideByVector(s1);
	s1.transpose();
	xy2.eltwiseDivideByVector(s1);

	//printf("3\n");
	///// part3
	// magnitude
	data3.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);
	s1.apply(NVMatrixOps::Sqrt());
	
	// cross correlation
	NVMatrix xy3;
	data3.transpose(xx);
	xx.rightMult(data3, xy3);
    
	// normlization
	s1.transpose();
	xy3.eltwiseDivideByVector(s1);
	s1.transpose();
	xy3.eltwiseDivideByVector(s1);

	//printf("4\n");
	////// fusion
	xy1.add(xy2, alpha, beta);
	xy1.add(xy3, gamma);

	// binomial deviance
	xy1.addScalar(-0.5f);
	xy1.eltwiseMult(M);
	xy1.apply(NVMatrixOps::Deviance(3.0f));
	xy1.eltwiseMult(W);

	return xy1.sum();
}

void computeCosine5Grad(NVMatrix& labels, NVMatrix& data1, NVMatrix& data2, NVMatrix& data3, 
	float alpha, float beta, float gamma,
	NVMatrix& target1, NVMatrix& target2, NVMatrix& target3, 
	bool add, float coeff)
{
	int numCases = data1.getLeadingDim(); 
    int numOut = data1.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(data1.isContiguous());
    assert(labels.isContiguous());
    assert(!data1.isTrans());
	assert(!labels.isTrans());

	// labels to mask
	NVMatrix M(numCases, numCases);
	NVMatrix W(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0f / (2 * n1), 1.0f / (2 * n2), W.getDevData());

	////// part1
	// XTX 
	NVMatrix xx, yy, xtx1;
	data1.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, xtx1);

	// magnitude
	xtx1.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	NVMatrix E1;
	data1.transpose(xx);
	xx.rightMult(data1, E1);

	// normlization
	xtx1.transpose();		// i
	E1.eltwiseDivideByVector(xtx1);
	xtx1.transpose();		// j
	E1.eltwiseDivideByVector(xtx1);

	////// part2
	// XTX 
	NVMatrix xtx2;
	data2.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, xtx2);

	// magnitude
	xtx2.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	NVMatrix E2;
	data2.transpose(xx);
	xx.rightMult(data2, E2);

	// normlization
	xtx2.transpose();		// i
	E2.eltwiseDivideByVector(xtx2);
	xtx2.transpose();		// j
	E2.eltwiseDivideByVector(xtx2);

	////// part3
	// XTX 
	NVMatrix xtx3;
	data3.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, xtx3);

	// magnitude
	xtx3.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	NVMatrix E3;
	data3.transpose(xx);
	xx.rightMult(data3, E3);

	// normlization
	xtx3.transpose();		// i
	E3.eltwiseDivideByVector(xtx3);
	xtx3.transpose();		// j
	E3.eltwiseDivideByVector(xtx3);

	////// fusion
	E1.add(E2, alpha, beta);
	E1.add(E3, gamma);

	E1.copy(E2);
	E1.copy(E3);

	////// part1
	// V1, V2
	NVMatrix V1, V2;
	
	xtx1.transpose();		// i
	E1.eltwiseDivideByVector(xtx1, V1);
	V1.eltwiseDivideByVector(xtx1);

	xtx1.transpose();		// j
	E1.eltwiseDivideByVector(xtx1, V2);
	V2.eltwiseDivideByVector(xtx1);
    
	// E
	// binomial deviance
	E1.addScalar(-0.5f);
	E1.eltwiseMult(M);
	E1.apply(NVMatrixOps::Logistic(3.0f));
	E1.eltwiseMult(M);
	E1.eltwiseMult(W); 
	E1.scale(3.0f * coeff * alpha);

	// update EU, EV
	V1.eltwiseMult(E1);
	V2.eltwiseMult(E1);

	E1.eltwiseDivideByVector(xtx1);
	xtx1.transpose();		// i
	E1.eltwiseDivideByVector(xtx1);

	// G_X
	V1.sum(1, xtx1);
	xtx1.transpose();
	data1.eltwiseMultByVector(xtx1, xx);

	V2.sum(0, xtx1);
	data1.eltwiseMultByVector(xtx1, yy);
	xx.add(yy);

	data1.rightMult(E1, yy);
	xx.add(yy, -1.0f, 1.0f);

	E1.transpose();
	data1.rightMult(E1, yy);
	xx.add(yy);

	if (!add) {
		xx.copy(target1);
	} else {
		target1.add(xx);
	}

	////// part2
	// V1, V2
	xtx2.transpose();		// i
	E2.eltwiseDivideByVector(xtx2, V1);
	V1.eltwiseDivideByVector(xtx2);

	xtx2.transpose();		// j
	E2.eltwiseDivideByVector(xtx2, V2);
	V2.eltwiseDivideByVector(xtx2);
    
	// E
	// binomial deviance
	E2.addScalar(-0.5f);
	E2.eltwiseMult(M);
	E2.apply(NVMatrixOps::Logistic(3.0f));
	E2.eltwiseMult(M);
	E2.eltwiseMult(W); 
	E2.scale(3.0f * coeff * beta);

	// update EU, EV
	V1.eltwiseMult(E2);
	V2.eltwiseMult(E2);

	E2.eltwiseDivideByVector(xtx2);
	xtx2.transpose();		// i
	E2.eltwiseDivideByVector(xtx2);

	// G_X
	V1.sum(1, xtx2);
	xtx2.transpose();
	data2.eltwiseMultByVector(xtx2, xx);

	V2.sum(0, xtx2);
	data2.eltwiseMultByVector(xtx2, yy);
	xx.add(yy);

	data2.rightMult(E2, yy);
	xx.add(yy, -1.0f, 1.0f);

	E2.transpose();
	data2.rightMult(E2, yy);
	xx.add(yy);

	if (!add) {
		xx.copy(target2);
	} else {
		target2.add(xx);
	}

	////// part3
	// V1, V2
	xtx3.transpose();		// i
	E3.eltwiseDivideByVector(xtx3, V1);
	V1.eltwiseDivideByVector(xtx3);

	xtx3.transpose();		// j
	E3.eltwiseDivideByVector(xtx3, V2);
	V2.eltwiseDivideByVector(xtx3);
    
	// E
	// binomial deviance
	E3.addScalar(-0.5f);
	E3.eltwiseMult(M);
	E3.apply(NVMatrixOps::Logistic(3.0f));
	E3.eltwiseMult(M);
	E3.eltwiseMult(W); 
	E3.scale(3.0 * coeff * gamma);

	// update EU, EV
	V1.eltwiseMult(E3);
	V2.eltwiseMult(E3);

	E3.eltwiseDivideByVector(xtx3);
	xtx3.transpose();		// i
	E3.eltwiseDivideByVector(xtx3);

	// G_X
	V1.sum(1, xtx3);
	xtx3.transpose();
	data3.eltwiseMultByVector(xtx3, xx);

	V2.sum(0, xtx3);
	data3.eltwiseMultByVector(xtx3, yy);
	xx.add(yy);

	data3.rightMult(E3, yy);
	xx.add(yy, -1.0f, 1.0f);

	E3.transpose();
	data3.rightMult(E3, yy);
	xx.add(yy);

	if (!add) {
		xx.copy(target3);
	} else {
		target3.add(xx);
	}
}

/*
 * age cost: L1 = square(predict[0] - age)
 * gender cost: L2 = log(1 + exp(-2 * predict[1] * gender))
 * race cost: L3 = log(1 + exp(-2 * predict[2] * race))
 *
 * age:         (1, numCases)
 * gender:      (1, numCases)
 * race:        (1, numCases)
 * predict:		(3, numCases)
 * output:		(1, numCases)   (*out)
 */
void computeAGRCost(NVMatrix& age, NVMatrix& gender, NVMatrix& race, NVMatrix& predict, 
	NVMatrix& ageLoss, NVMatrix& genderLoss, NVMatrix& raceLoss, NVMatrix& output)
{
	// age, gender, and race loss
	predict.sliceRows(0, 1, ageLoss);
	predict.sliceRows(1, 2, genderLoss);
	predict.sliceRows(2, 3, raceLoss);

	ageLoss.squaredDiff(age);

	genderLoss.eltwiseMult(gender);
	genderLoss.apply(NVMatrixOps::Deviance(2.0f));

	raceLoss.eltwiseMult(race);
	raceLoss.apply(NVMatrixOps::Deviance(2.0f));

	// merge loss
	ageLoss.copy(output);
	output.add(genderLoss);
	output.add(raceLoss);
}

/*
 * age cost: L1 = square(predict[0] - age)
 * gender cost: L2 = log(1 + exp(-2 * predict[1] * gender))
 * race cost: L3 = log(1 + exp(-2 * predict[2] * race))
 * 
 * G1 = 2 * (predict[0] - age)
 * G2 = Logistic2(predict[1] * gender)
 */
void computeAGRGrad(NVMatrix& age, NVMatrix& gender, NVMatrix& race, NVMatrix& predict, NVMatrix& target, 
	bool add, float coeff)
{
	int numCases = predict.getLeadingDim(); 
    int numOut = predict.getFollowingDim(); 

	assert(predict.isContiguous());
    assert(!predict.isTrans());
    
    if (!add) {
        target.resize(predict);
		target.apply(NVMatrixOps::Zero());
    }

	// get predict
	NVMatrix ageP, genderP, raceP;
	predict.sliceRows(0, 1, ageP);
	predict.sliceRows(1, 2, genderP);
	predict.sliceRows(2, 3, raceP);

	// age
	ageP.subtract(age);
	ageP.apply(NVMatrixOps::MultByScalar(-2 * coeff));

	// gender
	genderP.eltwiseMult(gender);
	genderP.apply(NVMatrixOps::Logistic(2.0f));
	genderP.eltwiseMult(gender);
	genderP.apply(NVMatrixOps::MultByScalar(2 * coeff));

	// race
	raceP.eltwiseMult(race);
	raceP.apply(NVMatrixOps::Logistic(2.0f));
	raceP.eltwiseMult(race);
	raceP.apply(NVMatrixOps::MultByScalar(2 * coeff));

	// copy result
	NVMatrix ageGrad(target.getCellPtr(0, 0), 1, target.getNumCols(), -1, target.isTrans());
	NVMatrix genderGrad(target.getCellPtr(1, 0), 1, target.getNumCols(), -1, target.isTrans());
	NVMatrix raceGrad(target.getCellPtr(2, 0), 1, target.getNumCols(), -1, target.isTrans());

	if (!add) {
		ageP.copy(ageGrad);
		genderP.copy(genderGrad);
		raceP.copy(raceGrad);
	} else {
		ageGrad.add(ageP);
		genderGrad.add(genderP);
		raceGrad.add(raceP);
	}
}

/*
 * attr cost: log(1 + exp(-predict * attr))
 *
 * attr:        (21, numCases)
 * predict:		(21, numCases)
 * output:		(1, numCases)   (*out)
 */
void computeAttrCost(NVMatrix& attr, NVMatrix& predict, NVMatrix& output)
{
	int numAttr = attr.getFollowingDim();

	NVMatrix loss;

	attr.eltwiseMult(predict, loss);
	loss.apply(NVMatrixOps::Deviance(1.0f));

	loss.sum(0, output);
	output.scale(1.0f / numAttr);
}

/*
 * attr cost: log(1 + exp(-predict * attr))
 * 
 * gradient: attr * Logistic(predict * attr)
 */
void computeAttrGrad(NVMatrix& attr, NVMatrix& predict, NVMatrix& target, 
	bool add, float coeff)
{
	int numCases = predict.getLeadingDim(); 
    int numAttr = predict.getFollowingDim(); 

	assert(predict.isContiguous());
    assert(!predict.isTrans());
    
	NVMatrix grad;
	// attr
	attr.eltwiseMult(predict, grad);
	grad.apply(NVMatrixOps::Logistic(1.0f));
	grad.eltwiseMult(attr);
	grad.apply(NVMatrixOps::MultByScalar(coeff));
	grad.scale(1.0f / numAttr);

	if (!add) {
		//printf("attr: 0\n");
		grad.copy(target);
	} else {
		//printf("attr: 1\n");
		target.add(grad);
	}
}


float computeFisherCost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe) {
    int numCases = gallery.getNumCols(); 
    int numD = gallery.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
    assert(labels.isContiguous());
    assert(gallery.isContiguous());
	assert(probe.isContiguous());

	// labels to mask
	NVMatrix M(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	float diff = M.sum();
	float w1 = 2.0f / (numCases * numCases + diff);
	float w2 = 2.0f / (numCases * numCases - diff);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, w1, -w2, M.getDevData());

	// square: s1 = X^2, s2 = Y^2
	NVMatrix s, s1, s2;
	gallery.apply(NVMatrixOps::Square(), s);
	s.sum(0, s1);
	s1.transpose();

	probe.apply(NVMatrixOps::Square(), s);
	s.sum(0, s2);

	// cross: E = -2 * XT * Y
	NVMatrix E;
	gallery.transpose();
	gallery.rightMult(probe, -2.0f, E);
	gallery.transpose();

	// distance: E = X^2 + Y^2 - 2 * XT * Y
	E.addVector(s1);
	E.addVector(s2);

	// between class divergence: (sum(E .* M))^2
	M.eltwiseMult(E);
	float a = M.sum();

	// total divergence: sum((E - mean(E))^2)
	float m = E.mean();
	E.addScalar(-m);
	float b = E.norm2();

	return -a * a * (numCases * numCases) / b;
}

void computeFisherGrad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, 
	NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff)
{
	int numCases = gallery.getLeadingDim(); 
    int numD = gallery.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(gallery.isContiguous());
    assert(probe.isContiguous());
    assert(labels.isContiguous());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
	assert(!labels.isTrans());
    
    if (!add) {
        galleryTarget.resize(gallery);
        probeTarget.resize(probe);
    }

	// labels to mask
	NVMatrix M(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	float diff = M.sum();
	float w1 = 2.0f / (numCases * numCases + diff);
	float w2 = 2.0f / (numCases * numCases - diff);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, w1, -w2, M.getDevData());
	
	NVMatrix tm, te;				// nx1
	NVMatrix xx, yy;				// dxn

	// square: s1 = X^2, s2 = Y^2
	NVMatrix s1, s2;
	gallery.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);
	s1.transpose();

	probe.apply(NVMatrixOps::Square(), yy);
	yy.sum(0, s2);

	// cross: E = -2 * XT * Y
	NVMatrix E;
	gallery.transpose();
	gallery.rightMult(probe, -2.0f, E);
	gallery.transpose();

	// distance: E = X^2 + Y^2 - 2 * XT * Y
	E.addVector(s1);
	E.addVector(s2);

	// factors
	NVMatrix E1;
	float meanE = E.mean();
	E.addScalar(-meanE, E1);
	float magE = E1.norm2();

	E.eltwiseMult(M, E1);
	float a = E1.sum() / magE;

	// 
	M.add(E, -4.0f * a, 4.0f * a * a, E1);
	E1.addScalar(-4.0f * a * a * meanE);

	// e~, m~
	M.sum(1, tm);
	tm.transpose();
	E.sum(1, te);
	te.transpose();
	tm.add(te, 4.0f * a, -4.0f * a * a);
	tm.addScalar(4.0f * a * a * meanE * numCases);

	// G_X
	gallery.eltwiseMultByVector(tm, xx);
	E1.transpose();
	probe.rightMult(E1, yy);
	E1.transpose();
	xx.add(yy);

	xx.scale(coeff * numCases * numCases);

	if (!add) {
		xx.copy(galleryTarget);
	} else {
		galleryTarget.add(xx);
	}

	// e^, m^
	M.sum(0, tm);
	E.sum(0, te);
	tm.add(te, 4.0f * a, -4.0f * a * a);
	tm.addScalar(4.0f * a * a * meanE * numCases);

	// G1_Y
	probe.eltwiseMultByVector(tm, yy);
	gallery.rightMult(E1, xx);
	yy.add(xx);

	yy.scale(coeff * numCases * numCases);

	// G_Y
	if (!add) {
		yy.copy(probeTarget);
	} else {
		probeTarget.add(yy);
	}
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

float computeKnifeCost(NVMatrix& labels, NVMatrix& data) {
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
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, -1.0f, 1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");

	// W
	NVMatrix W(numCases, numCases);
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n - diff) / 2.0f;	// positive count
	float n2 = (n + diff) / 2.0f;	// negative count
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1 / (2 * n1), 1 / (2 * n2), W.getDevData());

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

	// weighted mean
	E.eltwiseMult(W, s);
	float m = s.sum();
	E.addScalar(-m);
	//printf("threshold: %f\n", m);

	// deviance
	E.eltwiseMult(M);
	E.apply(NVMatrixOps::Deviance(1.0f));
	E.eltwiseMult(W);
	
	return E.sum();
}

void computeKnifeGrad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, bool add, float coeff)
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
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, -1.0f, 1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");

	// W, U, V
	NVMatrix W(numCases, numCases);
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n - diff) / 2.0f;	// positive count
	float n2 = (n + diff) / 2.0f;	// negative count
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1 / (2 * n1), 1 / (2 * n2), W.getDevData());

	NVMatrix tz, tw;			// nx1
	NVMatrix xx, yy;			// dxn

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

	// weighted mean
	E.eltwiseMult(W, s1);
	float m = s1.sum();
	E.addScalar(-m);

	// Deviance: Z
	//printf("1\n");
	E.eltwiseMult(M);
	E.apply(NVMatrixOps::Logistic(1.0f));
	E.eltwiseMult(M);
	E.eltwiseMult(W);
	E.scale(2.0f * coeff);

	// Mean term
	// tw~
	W.sum(1, tw);
	tw.transpose();
	data.eltwiseMultByVector(tw, xx);

	W.sum(0, tw);
	data.eltwiseMultByVector(tw, s1);
	xx.add(s1);

	//printf("2.1\n");
	data.rightMult(W, s1);
	xx.add(s1, -1.0f);

	//printf("2.2\n");
	W.transpose();
	data.rightMult(W, s1);
	xx.add(s1, -1.0f);

	//printf("2.3\n");
	xx.scale(E.sum());

	//printf("2.4\n");

	// G_X
	// tz~ 
	E.sum(1, tz);
	tz.transpose();
	data.eltwiseMultByVector(tz, yy);

	E.sum(0, tz);
	data.eltwiseMultByVector(tz, s1);
	yy.add(s1);

	data.rightMult(E, s1);
	yy.add(s1, -1.0f);

	E.transpose();
	data.rightMult(E, s1);
	yy.add(s1, -1.0f);

	yy.add(xx, -1.0f);
	
	if (!add) {
		yy.copy(target);
	} else {
		target.add(yy);
	}
}

float computeKnife2Cost(NVMatrix& labels, NVMatrix& data) {
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
	float n1 = (n + diff) / 2.0f;
	float n2 = (n - diff) / 2.0f;

	// weights
	NVMatrix W(numCases, numCases);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1.0 / (2 * n1), 1.0 / (2 * n2), W.getDevData());
	float sum = W.sum();
	W.scale(1.0f / sum);

	// magnitude
	NVMatrix xx, s;
	data.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s);
	s.apply(NVMatrixOps::Sqrt());

	// cross correlation
	NVMatrix E;
	data.copy(xx);
	xx.transpose();
	xx.rightMult(data, E);
    
	// normlization
	s.transpose();	// i
	E.eltwiseDivideByVector(s);
	s.transpose();	// j
	E.eltwiseDivideByVector(s);

	// weighted mean
	E.eltwiseMult(W, s);
	float m = s.sum();
	E.addScalar(-m);
	//printf("threshold: %f\n", m);

	// deviance
	E.eltwiseMult(M);
	E.apply(NVMatrixOps::Deviance(3.0f));
	E.eltwiseMult(W);
	
	return E.sum();
}

void computeKnife2Grad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, bool add, float coeff)
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

	// W, U, V
	NVMatrix W(numCases, numCases);
	float diff = M.sum();
	float n = numCases * (numCases - 1.0f) / 2.0f;
	float n1 = (n + diff) / 2.0f;	// positive count
	float n2 = (n - diff) / 2.0f;	// negative count
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, true, 1 / (2 * n1), 1 / (2 * n2), W.getDevData());

	NVMatrix wu, wv;			// nx1
	NVMatrix xx, yy;			// dxn

	// square: s1 = X^2
	NVMatrix s1;
	data.apply(NVMatrixOps::Square(), xx);
	xx.sum(0, s1);

	// cross: E = XT * X
	NVMatrix E;
	data.copy(xx);
	xx.transpose();
	xx.rightMult(data, E);

	// normlization
	s1.transpose();	// i
	E.eltwiseDivideByVector(s1);
	s1.transpose();	// j
	E.eltwiseDivideByVector(s1);

	// U, V1, V2
	NVMatrix U(numCases, numCases), V1, V2;

	s1.transpose();	// i
	E.eltwiseDivideByVector(s1, V1);
	V1.eltwiseDivideByVector(s1);

	s1.transpose();	// j
	E.eltwiseDivideByVector(s1, V2);
	V2.eltwiseDivideByVector(s1);
	
	U.apply(NVMatrixOps::One());
	U.eltwiseDivideByVector(s1);
	s1.transpose();	// i
	U.eltwiseDivideByVector(s1);

	// weighted mean
	E.eltwiseMult(W, s1);
	float m = s1.sum();
	E.addScalar(-m);

	// Deviance: Z
	E.eltwiseMult(M);
	E.apply(NVMatrixOps::Logistic(3.0f));
	E.eltwiseMult(M);
	E.eltwiseMult(W);
	E.scale(3.0f * coeff);

	// WV1, WV2, WU + WU'
	NVMatrix WV1, WV2, WU;

	W.eltwiseMult(V1, WV1);
	W.eltwiseMult(V2, WV2);
	W.eltwiseMult(U, WU);

	WU.transpose(M);
	WU.add(M);

	// Mean term
	// wv~
	WV1.sum(1, wv);
	wv.transpose();
	data.eltwiseMultByVector(wv, xx);

	WV2.sum(0, wv);
	data.eltwiseMultByVector(wv, s1);
	xx.add(s1);

	data.rightMult(WU, s1);
	xx.add(s1, -1.0f, 1.0f);

	xx.scale(E.sum());

	// ZV1, ZV2, ZU + ZU'
	E.eltwiseMult(V1, WV1);
	E.eltwiseMult(V2, WV2);
	E.eltwiseMult(U, WU);

	WU.transpose(M);
	WU.add(M);

	// G_X
	// zv~
	WV1.sum(1, wv);
	wv.transpose();
	data.eltwiseMultByVector(wv, yy);

	WV2.sum(0, wv);
	data.eltwiseMultByVector(wv, s1);
	yy.add(s1);

	data.rightMult(WU, s1);
	yy.add(s1, -1.0f, 1.0f);

	yy.add(xx, -1.0f);
	
	if (!add) {
		yy.copy(target);
	} else {
		target.add(yy);
	}
}

float computeDPCost(NVMatrix& labels, NVMatrix& X, NVMatrix& Y)
{
	int numCases = X.getNumCols(); 
    int numD = X.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!X.isTrans());
	assert(!Y.isTrans());
    assert(labels.isContiguous());
    assert(X.isContiguous());
	assert(Y.isContiguous());

	// labels to mask
	NVMatrix M(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	float diff = M.sum();
	float w1 = 2.0f / (numCases * numCases + diff);
	float w2 = 2.0f / (numCases * numCases - diff);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, w1, -w2, M.getDevData());

	// cross: E = XT * Y
	NVMatrix s, E;
	X.copy(s);
	s.transpose();
	s.rightMult(Y, E);

	// between class divergence: (sum(E .* M))^2
	M.eltwiseMult(E);
	float a = M.sum();

	// total divergence: sum((E - mean(E))^2)
	float m = E.mean();
	E.addScalar(-m);
	float b = E.norm2();

	return -a * a * numCases * numCases / b;
}


void computeDPGrad(NVMatrix& labels, NVMatrix& X, NVMatrix& Y, NVMatrix& xTarget, NVMatrix& yTarget, 
	bool add, float coeff)
{
	int numCases = X.getLeadingDim(); 
    int numD = X.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(X.isContiguous());
    assert(labels.isContiguous());
    assert(!X.isTrans());
	assert(!labels.isTrans());

	// labels to mask
	NVMatrix M(numCases, numCases);
	const int BLOCK_DIM_X = 32;
	const int BLOCK_DIM_Y = 32;
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
	dim3 gridDim((numCases + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numCases + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, 1.0f, -1.0f, M.getDevData());
    getLastCudaError ("kLabel2Weight: Kernel execution failed");
	float diff = M.sum();
	float w1 = 2.0f / (numCases * numCases + diff);
	float w2 = 2.0f / (numCases * numCases - diff);
	kLabel2Weight<<<gridDim, blockDim>>>(labels.getDevData(), numCases, false, w1, -w2, M.getDevData());
	
	NVMatrix xx, yy;				// dxn

	// cross: E = XT * Y
	NVMatrix E;
	X.copy(xx);
	xx.transpose();
	xx.rightMult(Y, E);

	// factors
	NVMatrix E1;
	float meanE = E.mean();
	E.addScalar(-meanE, E1);
	float magE = E1.norm2();

	E.eltwiseMult(M, E1);
	float a = E1.sum() / magE;

	// 
	M.add(E, 2.0f * a, -2.0f * a * a, E1);
	E1.addScalar(2.0f * a * a * meanE);

	// G_X
	E1.transpose();
	Y.rightMult(E1, xx);
	xx.scale(coeff * numCases * numCases);
	xx.flipTrans(xTarget);

	// G_Y
	E1.transpose();
	X.rightMult(E1, yy);
	yy.scale(coeff * numCases * numCases);
	yy.flipTrans(yTarget);
}

float computeDP2Cost(NVMatrix& labels, NVMatrix& X)
{
	int numCases = X.getNumCols(); 
    int numD = X.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!X.isTrans());
    assert(labels.isContiguous());
    assert(X.isContiguous());

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

	// cross: E = XT * X
	NVMatrix s, E;
	X.copy(s);
	s.transpose();
	s.rightMult(X, E);

	// between class divergence: (sum(E .* M))^2
	M.eltwiseMult(E);
	float a = M.sum();

	// total divergence: sum((E - mean(E))^2)
	float m = E.mean();
	E.addScalar(-m);
	float b = E.norm2();

	return -a * a * n / b;
}

void computeDP2Grad(NVMatrix& labels, NVMatrix& X, NVMatrix& target, 
	bool add, float coeff)
{
	int numCases = X.getLeadingDim(); 
    int numD = X.getFollowingDim(); 

    assert(labels.getNumElements() == numCases);
    assert(X.isContiguous());
    assert(labels.isContiguous());
    assert(!X.isTrans());
	assert(!labels.isTrans());
    
    if (!add) {
        target.resize(X);
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
	
	NVMatrix xx, yy;				// dxn

	// cross: E = XT * X
	NVMatrix E;
	X.copy(xx);
	xx.transpose();
	xx.rightMult(X, E);

	// factors
	NVMatrix E1;
	float meanE = E.mean();
	E.addScalar(-meanE, E1);
	float magE = E1.norm2();

	E.eltwiseMult(M, E1);
	float a = E1.sum() / magE;

	// 
	M.add(E, 2.0f * a, -2.0f * a * a, E1);
	E1.addScalar(2.0f * a * a * meanE);

	// G_X
	X.rightMult(E1, xx);
	E1.transpose();
	X.rightMult(E1, yy);
	xx.add(yy);
	xx.scale(coeff * n);

	xx.flipTrans(target);
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
* L2Cost:
* Loss = 0.5 * (gallery - probe) ^ 2, if label = 1;
*        0.5 * [m - |gallery - probe|2] ^ 2, if label = -1.
*/
void computeL2Cost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& output, float m) {
    int numCases = gallery.getNumCols(); 
    int numD = gallery.getNumRows(); 
	//float m = 0;

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
    assert(labels.isContiguous());
    assert(gallery.isContiguous());
	assert(probe.isContiguous());

	output.resize(labels);

	// calculate L2-norm of all pair samples: dists
	NVMatrix s1(gallery, true), s2(probe, true), temp(gallery, true), dists(labels, true);
	//s1.print(160,5);
	//s2.print(160,5);
	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	temp.sum(0, dists);
	dists.apply(NVMatrixOps::Sqrt());
	//dists.print(1,128);

	// calculate the optimal threshold: m
	/*Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
	dists.copyToHost(devData);
	labels.copyToHost(devLabels);
	m = getOptimalThr(devData, devLabels, numCases);*/
	//m = 3;

	// loss of positive samples: posLoss
	NVMatrix posMask(labels, true), posLoss(labels, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(0.5, 0.5), posMask);
	//posMask.print(1,128); //debug
	dists.apply(NVMatrixOps::Square(), posLoss);
	posLoss.apply(NVMatrixOps::MultByScalar(0.5));
	posLoss.eltwiseMult(posMask);
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
	//negLoss.print(1,128);  //debug

	// add posLoss and negLoss: output
	posLoss.add(negLoss, output);
	//output.print(1,128);  //debug
}

/**
* L2Cost:
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
void computeL2Grad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, float m,
	bool add, float coeff) {

    int numCases = gallery.getLeadingDim(); 
    int numOut = gallery.getFollowingDim(); 
	//float m = 0;

    assert(labels.getNumElements() == numCases);
    assert(gallery.isContiguous());
    assert(probe.isContiguous());
    assert(labels.isContiguous());
    assert(!gallery.isTrans());
	assert(!probe.isTrans());
	assert(!labels.isTrans());

   
    if (!add) {
        galleryTarget.resize(gallery);
        probeTarget.resize(probe);
    }

	// calculate L2-norm of all pair samples: dists
	NVMatrix s1(gallery, true), s2(probe, true), temp(gallery, true), dists(labels, true);
	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	temp.sum(0, dists);
	dists.apply(NVMatrixOps::Sqrt());
	
	// calculate the optimal threshold: m
	/*Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
	dists.copyToHost(devData);
	labels.copyToHost(devLabels);
	m = getOptimalThr(devData, devLabels, numCases);*/
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
	dists.apply(NVMatrixOps::MultByScalar(m * coeff));
	dists.apply(NVMatrixOps::AddScalar(-1.0));
	diff1.eltwiseMultByVector(dists, G_X_n);
	diff2.eltwiseMultByVector(dists, G_Y_n);
	G_X_n.eltwiseMultByVector(negMask1);
	G_X_n.eltwiseMultByVector(negMask2);
	G_Y_n.eltwiseMultByVector(negMask1);
	G_Y_n.eltwiseMultByVector(negMask2);
	//G_Y_n.print(160,5);

	// G_X = G_X_p + G_X_n, G_Y = G_Y_p + G_Y_n
	if (!add) {
		G_X_p.add(G_X_n, galleryTarget);
		G_Y_p.add(G_Y_n, probeTarget);
	} else {
		galleryTarget.add(G_X_p);
		galleryTarget.add(G_X_n);
		probeTarget.add(G_Y_p);
		probeTarget.add(G_Y_n);
	}
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
 * hinge: L = max(0, 1 - cosine(gallery, probe) * labels)  -- cannot work
 * square: L = square(cosine(gallery, probe) - labels)
 * exp: L = exp(-cosine(gallery, probe) * labels)
 * binomial deviance: L = log(1 + exp(-2 * cosine(gallery, probe) * labels))
 * huber: L = Huber(cosine(gallery, probe) * labels - 1) 
 * 
 * labels:  (1, numCases)
 * gallery: (numD, numCases)
 * probe:   (numD, numCases) 
 * output:  (1, numCases)   (*out)
 */
void computeCosineSNCost(NVMatrix& idens, NVMatrix& data, NVMatrix& output) {
    int numCases = data.getNumCols(); 
    int numD = data.getNumRows(); 

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

	// magnitude
	NVMatrix s1(gallery, true), s2(probe, true);
	s1.apply(NVMatrixOps::Square());
	s2.apply(NVMatrixOps::Square());
	NVMatrix& magXY = s1.sum(0);
	s2.sum(0, output);
	magXY.eltwiseMult(output);
	magXY.apply(NVMatrixOps::Sqrt());

	// cross correlation
	gallery.copy(s1);
	s1.eltwiseMult(probe);
	s1.sum(0, output);
    
	// normlization
	output.eltwiseDivide(magXY);
	delete &magXY;

	// hinge loss
	/*output.eltwiseMult(labels);
	output.apply(NVMatrixOps::MultByScalar(-1));
	output.apply(NVMatrixOps::AddScalar(1));
	output.maxWithScalar(0);*/

	// squared loss
	// output.applyBinary(NVMatrixBinaryOps::SquaredDiff(), labels);

	// exp loss
	/*output.eltwiseMult(labels);
	output.apply(NVMatrixOps::MultByScalar(-1));
	output.apply(NVMatrixOps::Exp());*/

	// binomial deviance
	output.eltwiseMult(labels);
	output.apply(NVMatrixOps::Deviance(2.0f));
}

/**
 * L = max(0, 1 - cosine(gallery, probe) * labels) -- cannot work
 * L = square(cosine(gallery, probe) - labels)
 * L = exp(-cosine(gallery, probe) * labels)
 * L = log(1 + exp(-2 * cosine(gallery, probe) * labels))
 * L = huber(cosine(gallery, probe) * labels) 
 * 
 * L(X, Y, Z) = ||X' * Y / (sqrt(X' * X) * sqrt(Y' * Y)) - Z||
 * G_X = 2 * L * G1_X
 * G_Y = 2 * L * G1_Y
 * G1_X = Y / magXY - X' * Y * X / (X' * X * magXY)
 * G1_Y = X / magXY - X' * Y * Y / (Y' * Y * magXY)
 * magXY = sqrt(X' * X) * sqrt(Y' * Y)
 */
void computeCosineSNGrad(NVMatrix& idens, NVMatrix& data, NVMatrix& dataTarget, 
	bool add, float coeff) {

    int numCases = data.getLeadingDim(); 
    int numOut = data.getFollowingDim(); 

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

	// XTX, YTY 
	NVMatrix temp1(gallery, true), temp2(probe, true);
	temp1.apply(NVMatrixOps::Square());
	temp2.apply(NVMatrixOps::Square());
	NVMatrix& XTX = temp1.sum(0);
	NVMatrix& YTY = temp2.sum(0);

	// magnitude
	NVMatrix magXY(XTX, true);
	magXY.eltwiseMult(YTY);
	magXY.apply(NVMatrixOps::Sqrt());

	// XTY: cross correlation
	gallery.copy(temp1);
	temp1.eltwiseMult(probe);
	NVMatrix& XTY = temp1.sum(0);
    
	// L: normlization
	NVMatrix L(XTY, true);
	L.eltwiseDivide(magXY);
	
	// hinge loss
	/*labels.copy(L);
	L.apply(NVMatrixOps::MultByScalar(coeff));*/

	// square loss
	//L.subtract(labels);
	//L.apply(NVMatrixOps::MultByScalar(-2 * coeff));

	// exp loss
	/*L.eltwiseMult(labels);
	L.apply(NVMatrixOps::MultByScalar(-1));
	L.apply(NVMatrixOps::Exp());
	L.eltwiseMult(labels);
	L.apply(NVMatrixOps::MultByScalar(coeff));*/

	// binomial deviance
	L.eltwiseMult(labels);
	L.apply(NVMatrixOps::Logistic(2.0f));
	L.eltwiseMult(labels);
	L.apply(NVMatrixOps::MultByScalar(2 * coeff));

	// G1_X
	probe.copy(temp1);
	gallery.copy(temp2);
	temp2.eltwiseMultByVector(XTY);
	temp2.eltwiseDivideByVector(XTX);
	temp1.subtract(temp2);
	temp1.eltwiseDivideByVector(magXY);

	// G_X
	temp1.eltwiseMultByVector(L, galleryTarget);
	galleryTarget.copy(tempTarget, 0, -1, 0, -1, 0, 0);

	// G1_Y
	gallery.copy(temp1);
	probe.copy(temp2);
	temp2.eltwiseMultByVector(XTY);
	temp2.eltwiseDivideByVector(YTY);
	temp1.subtract(temp2);
	temp1.eltwiseDivideByVector(magXY);

	// G_Y
	temp1.eltwiseMultByVector(L, probeTarget);
	probeTarget.copy(tempTarget, 0, -1, 0, -1, 0, numCases/2);

    // dataTarget
    if (!add) {
        tempTarget.copy(dataTarget);
    } else {
        dataTarget.add(tempTarget);
    }

	delete &XTX;
	delete &YTY;
	delete &XTY;
}

/**
* L3SNCost:
* L2 with normalization
*/
void computeL3SNCost(NVMatrix& idens, NVMatrix& data, NVMatrix& output, float& m) {
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
	//s1.apply(NVMatrixOps::Square());
	//s2.apply(NVMatrixOps::Square());
	//NVMatrix& magG = s1.sum(0);
	//NVMatrix& magP = s2.sum(0);
	//magG.apply(NVMatrixOps::Abs());
	//magP.apply(NVMatrixOps::Abs());
	//magG.apply(NVMatrixOps::Sqrt());
	//magP.apply(NVMatrixOps::Sqrt());
	////magG.addScalar(0.000001);
	////magP.addScalar(0.000001);
	//gallery.copy(s1);
	//probe.copy(s2);
	//s1.eltwiseDivideByVector(magG);
	//s2.eltwiseDivideByVector(magP);
	//delete &magG;
	//delete &magP;

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

	// calculate the optimal threshold: m
	if(1)
	{
		Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
		dists.copyToHost(devData);
		labels.copyToHost(devLabels);
		m = getOptimalThr(devData, devLabels, numCases/2);
	}
	//printf("%.3f ",m);
	//m = 3;
}

/**
* L3SNCost:
* Loss = 0.5 * (gallery - probe) ^ 2, if label = 1;
*        0.5 * [m - (gallery - probe)^2], if label = -1.
* 
* If label = 1:
*    G_X = probe - gallery
*    G_Y = gallery - probe
* if label = -1:
*    G_X = gallery - probe or 0
*    G_Y = probe - gallery or 0
*/
void computeL3SNGrad(NVMatrix& idens, NVMatrix& data, NVMatrix& dataTarget, float& m,
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
	//s1.apply(NVMatrixOps::Square());
	//s2.apply(NVMatrixOps::Square());
	//NVMatrix& magG = s1.sum(0);
	//NVMatrix& magP = s2.sum(0);
	//magG.apply(NVMatrixOps::Abs());
	//magP.apply(NVMatrixOps::Abs());
	//magG.apply(NVMatrixOps::Sqrt());
	//magP.apply(NVMatrixOps::Sqrt());
	////magG.addScalar(0.000001);
	////magP.addScalar(0.000001);
	//gallery.copy(s1);
	//probe.copy(s2);
	//s1.eltwiseDivideByVector(magG);
	//s2.eltwiseDivideByVector(magP);

	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	temp.sum(0, dists);
	temp.apply(NVMatrixOps::Abs());
	dists.apply(NVMatrixOps::Sqrt());
	dists.apply(NVMatrixOps::Abs());
	//dists.addScalar(0.000001);

	// calculate the optimal threshold: m
	/*if(m<0)
	{
		Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
		dists.copyToHost(devData);
		labels.copyToHost(devLabels);
		m = getOptimalThr(devData, devLabels, numCases/2);
	}*/
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
	
	//gallery.copy(s1);
	//probe.copy(s2);
	//s1.apply(NVMatrixOps::Square());
	//s2.apply(NVMatrixOps::Square());
	//NVMatrix tmp1(magG, true), tmp2(magP, true);
	//magG.apply(NVMatrixOps::Cubic(),tmp1);
	//magP.apply(NVMatrixOps::Cubic(),tmp2);
	//magG.apply(NVMatrixOps::Reciprocal());
	//magP.apply(NVMatrixOps::Reciprocal());
	//tmp1.apply(NVMatrixOps::Reciprocal());
	//tmp2.apply(NVMatrixOps::Reciprocal());
	//s1.eltwiseMultByVector(tmp1, G_X_p);
	//s2.eltwiseMultByVector(tmp2, G_Y_p);
	//G_X_p.apply(NVMatrixOps::MultByScalar(-1.0));
	//G_Y_p.apply(NVMatrixOps::MultByScalar(-1.0));
	//s1.apply(NVMatrixOps::MultByScalar(-1.0));
	//s1.apply(NVMatrixOps::MultByScalar(-1.0));
	//G_X_p.addVector(magG);
	//G_Y_p.addVector(magP);
	//s1.addVector(magG);
	//s2.addVector(magP);
	//galleryTarget.eltwiseMult(G_X_p);
	//probeTarget.eltwiseMult(G_Y_p);
	//galleryTarget.eltwiseMult(s1);
	//probeTarget.eltwiseMult(s2);
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

	//delete &magG;
	//delete &magP;
}


/**
* Joint1Cost:
* Loss = 0.5 * (gallery - probe) ^ 2, if label = 1;
*        0.5 * [m - (gallery - probe)^2], if label = -1.
*/
void computeJoint1Cost(NVMatrix& idens, NVMatrix& data, NVMatrix& probs, NVMatrix& output, float m, float lambda) {
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
	//s1.print(160,5);
	//s2.print(160,5);
	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	//temp.print(160,5);
	temp.sum(0, dists);
	//dists.print(1,64);

	// calculate the optimal threshold: m
	/*if(!mDone)
	{
		Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
		dists.copyToHost(devData);
		labels.copyToHost(devLabels);
		m = getOptimalThr(devData, devLabels, numCases/2);
		mDone = true;
	}*/
	//m = 3;

	// loss of positive samples: posLoss
	NVMatrix posMask(labels, true), posLoss(labels, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(0.5, 0.5), posMask);
	//posMask.print(1,128); //debug
	//dists.apply(NVMatrixOps::Square(), posLoss);
	//posLoss.apply(NVMatrixOps::MultByScalar(0.5));
	dists.apply(NVMatrixOps::MultByScalar(0.5),posLoss);
	posLoss.eltwiseMult(posMask);
	//posLoss.print(1,128);  //debug

	// loss of negative samples: negLoss
	NVMatrix negMask1(labels, true), negMask2(labels, true), negLoss(labels, true);
	labels.apply(NVMatrixOps::WeightedAddScalar(-0.5, 0.5), negMask1);
	//negMask1.print(1,128);  //debug
	dists.apply(NVMatrixOps::SmallerThanScalar(m), negMask2);
	dists.apply(NVMatrixOps::WeightedAddScalar(-1.0, m), negLoss);
	//negLoss.apply(NVMatrixOps::Square());
	negLoss.apply(NVMatrixOps::MultByScalar(0.5));
	negLoss.eltwiseMult(negMask1);
	negLoss.eltwiseMult(negMask2);
	//negLoss.print(1,128);  //debug

	// add posLoss and negLoss: output
	posLoss.add(negLoss, output);
	//if(output.sum()<1)
	//	output.print(1,128);  //debug
}

/**
* Joint1Cost:
* Loss = 0.5 * (gallery - probe) ^ 2, if label = 1;
*        0.5 * [m - (gallery - probe)^2], if label = -1.
* 
* If label = 1:
*    G_X = probe - gallery
*    G_Y = gallery - probe
* if label = -1:
*    G_X = gallery - probe or 0
*    G_Y = probe - gallery or 0
*/
void computeJoint1Grad(NVMatrix& idens, NVMatrix& data, NVMatrix& probs, NVMatrix& dataTarget, NVMatrix& probsTarget, float m, float lambda,
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
	s1.applyBinary(NVMatrixBinaryOps::SquaredDiff(), s2, temp);
	temp.sum(0, dists);
	//dists.apply(NVMatrixOps::Sqrt());
	
	// calculate the optimal threshold: m
	/*Matrix devData(dists.getNumRows(), dists.getNumCols()), devLabels(labels.getNumRows(), labels.getNumCols());
	dists.copyToHost(devData);
	labels.copyToHost(devLabels);
	m = getOptimalThr(devData, devLabels, numCases/2);*/
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
	diff1.eltwiseMultByVector(negMask1, G_X_n);
	diff2.eltwiseMultByVector(negMask1, G_Y_n);
	G_X_n.eltwiseMultByVector(negMask2);
	G_Y_n.eltwiseMultByVector(negMask2);
	//G_Y_n.print(160,5);

	// G_X = G_X_p + G_X_n, G_Y = G_Y_p + G_Y_n
	G_X_p.add(G_X_n, galleryTarget);
    G_Y_p.add(G_Y_n, probeTarget);
	galleryTarget.copy(tempTarget, 0, -1, 0, -1, 0, 0);
	probeTarget.copy(tempTarget, 0, -1, 0, -1, 0, numCases/2);

	if (!add) {
		tempTarget.copy(dataTarget);
	} else {
		dataTarget.add(tempTarget);
	}
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