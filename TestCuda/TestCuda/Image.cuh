#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "cuda.h"
#include "cuda_runtime.h"

#include <CImg.h>


class Image
{
protected:
	unsigned char* _data;
	size_t _width, _height, _depth, _spectrum;
public:
	__host__ __device__ const unsigned char * data() const;
	__host__ __device__ size_t width() const;
	__host__ __device__ size_t height() const;
	__host__ __device__ size_t depth() const;
	__host__ __device__ size_t spectrum() const;
	__host__ __device__ size_t size() const;
	__host__ __device__ long offset(const int& x, const int& y, const int& z = 0, const int& c = 0) const;
	__host__ __device__ const unsigned char& at(const int& x, const int& y, const int& z = 0, const int& c = 0) const;
	__host__ __device__ unsigned char& at(const int& x, const int& y, const int& z = 0, const int& c = 0);
	__host__ __device__ double cubic_atXY(const double& fx, const double& fy, const int& z = 0, const int& c = 0) const;
};

class DeviceImage: public Image
{
public:
	DeviceImage(size_t width, size_t height, size_t depth, size_t spectrum);
	DeviceImage(const cimg_library::CImg<unsigned char>& image);
	~DeviceImage();
};

#endif

