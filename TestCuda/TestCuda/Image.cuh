#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "cuda.h"
#include "cuda_runtime.h"

#include <CImg.h>


class Image
{
public:
	unsigned char* data;
	int width, height, depth, spectrum;
	__host__ __device__ long offset(const int& x, const int& y, const int& z = 0, const int& c = 0) const;
	__host__ __device__ const unsigned char& at(const int& x, const int& y, const int& z = 0, const int& c = 0) const;
	__host__ __device__ unsigned char& at(const int& x, const int& y, const int& z = 0, const int& c = 0);
	__host__ __device__ double cubic_atXY(const double& fx, const double& fy, const int& z = 0, const int& c = 0) const;
};

__host__ Image* deviceImageFromCImg(const cimg_library::CImg<unsigned char>& image);

#endif

