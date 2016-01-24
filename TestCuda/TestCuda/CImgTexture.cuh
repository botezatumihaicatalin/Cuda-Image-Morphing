#pragma once

#include "CImg.h"
#include "cuda_runtime.h"
#include "cuda.h"

// Wrapper function over CImg.
// Creates a cudaTextureObject from a CImg, where each element is an uchar4.
class CImgTexture
{
private:
	cudaArray* cuArray;
public:
	cudaTextureObject_t tex;
	CImgTexture(const cimg_library::CImg<unsigned char>& image);
	~CImgTexture();

	__device__ uchar4 linearTex2D(float x, float y);
	__device__ uchar4 cubicTex2D(float x, float y);
};

