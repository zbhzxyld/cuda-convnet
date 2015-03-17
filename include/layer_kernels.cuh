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

#ifndef LAYER_KERNELS_CUH
#define	LAYER_KERNELS_CUH

#include <helper_cuda.h>
#include <nvmatrix.cuh>

#define LOGREG_GRAD_THREADS_X      32
#define LOGREG_GRAD_THREADS_Y      4

#define LOGREG_ERR_THREADS_X        128
#define LOGREG_ERR_THREADS_Y        1

#define COSINE_GRAD_THREADS_X      32
#define COSINE_GRAD_THREADS_Y      4

#define COSINE_ERR_THREADS_X        128
#define COSINE_ERR_THREADS_Y        1

void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add);
void computeCosineCost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& output);
float computeCosine2Cost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe);
float computeCosine3Cost(NVMatrix& labels, NVMatrix& data);
float computeCosine4Cost(NVMatrix& labels, NVMatrix& data, float alpha);
float computeCosine5Cost(NVMatrix& labels, NVMatrix& data1, NVMatrix& data2, NVMatrix& data3,
	float alpha, float beta, float gamma);
float computeFisherCost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe);
float computeFisher2Cost(NVMatrix& labels, NVMatrix& data, float alpha);
float computeKnifeCost(NVMatrix& labels, NVMatrix& data);
float computeKnife2Cost(NVMatrix& labels, NVMatrix& data);
float computeDPCost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe);
float computeDP2Cost(NVMatrix& labels, NVMatrix& data);
void computeAGRCost(NVMatrix& age, NVMatrix& gender, NVMatrix& race, NVMatrix& predict, 
	NVMatrix& ageLoss, NVMatrix& genderLoss, NVMatrix& raceLoss, NVMatrix& output);
void computeAttrCost(NVMatrix& attr, NVMatrix& predict, NVMatrix& output);
void computeL2Cost(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& output, float m);
void computeL2SNCost(NVMatrix& labels, NVMatrix& data, NVMatrix& output, float m);
void computeCosineSNCost(NVMatrix& labels, NVMatrix& data, NVMatrix& output);
void computeL3SNCost(NVMatrix& labels, NVMatrix& data, NVMatrix& output, float& m);
void computeJoint1Cost(NVMatrix& labels, NVMatrix& data, NVMatrix& probs, NVMatrix& output, float m, float lambda);
void computeL2regCost(NVMatrix& ground, NVMatrix& data, NVMatrix& output);



// Numerical stability optimization: this routine combines computeLogregGrad with computeSoftmaxGrad
// to avoi dividing and then multiplying by quantities that may be near zero.
void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add);
void computeCosineGrad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff);
void computeCosine2Grad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff);
void computeCosine3Grad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, 
	bool add, float coeff);
void computeCosine4Grad(NVMatrix& labels, NVMatrix& data, float alpha, NVMatrix& target, 
	bool add, float coeff);
void computeCosine5Grad(NVMatrix& labels, NVMatrix& data1, NVMatrix& data2, NVMatrix& data3, 
	float alpha, float beta, float gamma, 
	NVMatrix& target1, NVMatrix& target2, NVMatrix& target3, 
	bool add, float coeff);
void computeCCAGrad(NVMatrix& X, NVMatrix& Y, float lambda, NVMatrix& xTarget, NVMatrix& yTarget, 
	bool add, float coeff);
void computeFisherGrad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff);
void computeFisher2Grad(NVMatrix& labels, NVMatrix& data, float alpha, NVMatrix& target, 
	bool add, float coeff);
void computeKnifeGrad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, 
	bool add, float coeff);
void computeKnife2Grad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, 
	bool add, float coeff);
void computeDPGrad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, 
	bool add, float coeff);
void computeDP2Grad(NVMatrix& labels, NVMatrix& data, NVMatrix& target, 
	bool add, float coeff);
void computeAGRGrad(NVMatrix& age, NVMatrix& gender, NVMatrix& race, NVMatrix& predict, NVMatrix& target, 
	bool add, float coeff);
void computeAttrGrad(NVMatrix& attr, NVMatrix& predict, NVMatrix& target, 
	bool add, float coeff);
void computeL2Grad(NVMatrix& labels, NVMatrix& gallery, NVMatrix& probe, NVMatrix& galleryTarget, NVMatrix& probeTarget, float m,
	bool add, float coeff);
void computeL2SNGrad(NVMatrix& labels, NVMatrix& data, NVMatrix& dataTarget, float m,
	bool add, float coeff);
void computeCosineSNGrad(NVMatrix& labels, NVMatrix& data, NVMatrix& dataTarget, 
	bool add, float coeff);
void computeL3SNGrad(NVMatrix& labels, NVMatrix& data, NVMatrix& dataTarget, float& m,
	bool add, float coeff);
void computeJoint1Grad(NVMatrix& labels, NVMatrix& data, NVMatrix& probs, NVMatrix& dataTarget, NVMatrix& probsTarget, float m, float lambda,
	bool add, float coeff);
void computeL2regGrad(NVMatrix& ground, NVMatrix& data, NVMatrix& target, 
	bool add, float coeff);
#endif	/* LAYER_KERNELS_CUH */

