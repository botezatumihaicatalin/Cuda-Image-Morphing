#ifndef _MORPH_H_
#define _MORPH_H_

#include <vector>
#include <CImg.h>

#include "cuda_runtime.h"
#include "Delaunay.h"
#include "Image.cuh"

class DeviceMorph
{
private:
	DeviceImage *d_imageSrc, *d_imageDest, *d_output;
	DeviceImage *_imageSrc, *_imageDest, *_output;
	Point *d_pointsSrc, *d_pointsDest;

	IndexTriangle* d_triangles;
	size_t _trianglesSize;

	DeviceMorph* d_instance;

	__global__ friend void morphKernel(DeviceMorph* d_instance, double ratio = 1);
	__global__ friend void warpKernel(DeviceMorph* d_instance, double ratio = 1, int way = 1);

public:
	DeviceMorph(const cimg_library::CImg<unsigned char>& imageSrc,
	            const cimg_library::CImg<unsigned char>& imageDest,
	            const std::vector<Point>& pointsSrc, const std::vector<Point>& pointsDest,
	            const std::vector<IndexTriangle>& triangles);

	std::vector<cimg_library::CImg<unsigned char>> computeMorph() const;
	cimg_library::CImg<unsigned char> computeWarp(double ratio = 1, int way = 1) const;

	~DeviceMorph();
};

#endif

