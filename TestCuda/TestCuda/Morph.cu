#include "Morph.cuh"
#include "Image.cuh"

DeviceMorph::DeviceMorph(const cimg_library::CImg<unsigned char>& imageSrc, const cimg_library::CImg<unsigned char>& imageDest, const std::vector<Point>& pointsSrc, const std::vector<Point>& pointsDest, const std::vector<IndexTriangle>& triangles)
{
	if (!(imageSrc.width() == imageDest.width() &&
		imageSrc.height() == imageDest.height() &&
		imageSrc.depth() == imageDest.depth() &&
		imageSrc.spectrum() == imageDest.spectrum()))
	{
		throw std::invalid_argument("Image source must be same width/height/depth/spectrum as destination.");
	}

	for (size_t triangleIndex = 0; triangleIndex < triangles.size(); triangleIndex++)
	{
		const IndexTriangle& triangle = triangles[triangleIndex];
		for (size_t pIndex = 0; pIndex < 3; pIndex ++)
		{
			if (!(triangle.points[pIndex] < pointsSrc.size() && triangle.points[pIndex] < pointsDest.size()))
			{
				throw std::invalid_argument("Invalid triangulation for the given points.");
			}
		}
	}

	_imageSrcTexture = new CImgTexture(imageSrc);
	cudaMalloc(&d_imageSrcTexture, sizeof(CImgTexture));
	cudaMemcpy(d_imageSrcTexture, _imageSrcTexture, sizeof(CImgTexture), cudaMemcpyHostToDevice);

	_imageDestTexture = new CImgTexture(imageDest);
	cudaMalloc(&d_imageDestTexture, sizeof(CImgTexture));
	cudaMemcpy(d_imageDestTexture, _imageDestTexture, sizeof(CImgTexture), cudaMemcpyHostToDevice);

	_output = new DeviceImage(imageSrc);
	cudaMalloc(&d_output, sizeof(DeviceImage));
	cudaMemcpy(d_output, _output, sizeof(DeviceImage), cudaMemcpyHostToDevice);

	const Point* pointsSrcData = pointsSrc.data();
	cudaMalloc(&d_pointsSrc, sizeof(Point) * pointsSrc.size());
	cudaMemcpy(d_pointsSrc, pointsSrcData, sizeof(Point) * pointsSrc.size(), cudaMemcpyHostToDevice);

	const Point* pointsDestData = pointsDest.data();
	cudaMalloc(&d_pointsDest, sizeof(Point) * pointsDest.size());
	cudaMemcpy(d_pointsDest, pointsDestData, sizeof(Point) * pointsDest.size(), cudaMemcpyHostToDevice);

	const IndexTriangle* trianglesData = triangles.data();
	cudaMalloc(&d_triangles, sizeof(IndexTriangle) * triangles.size());
	cudaMemcpy(d_triangles, trianglesData, sizeof(IndexTriangle) * triangles.size(), cudaMemcpyHostToDevice);

	_trianglesSize = triangles.size();

	cudaMalloc(&d_instance, sizeof(DeviceMorph));
	cudaMemcpy(d_instance, this, sizeof(DeviceMorph), cudaMemcpyHostToDevice);
}

DeviceMorph::~DeviceMorph()
{
	cudaFree(d_pointsSrc);
	cudaFree(d_pointsDest);
	cudaFree(d_triangles);
	cudaFree(d_instance);

	delete _imageSrcTexture;
	delete _imageDestTexture;

	delete _output;
}

__host__ __device__ 
Point computePosition(Point& p, const Point* pointsSrc, const Point* pointsDest, const IndexTriangle* triangles, const size_t& trianglesSize, const double& ratio = 1)
{
	for (size_t trIdx = 0; trIdx < trianglesSize; trIdx++)
	{
		const Point& p1 = pointsDest[triangles[trIdx].points[0]];
		const Point& p2 = pointsDest[triangles[trIdx].points[1]];
		const Point& p3 = pointsDest[triangles[trIdx].points[2]];

		double bot = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
		double sTop = (p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y);
		double tTop = (p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y);

		double s = sTop / bot;
		double t = tTop / bot;

		if (!(s >= 0 && s <= 1 && t >= 0 && t <= 1 && (s + t) <= 1))
		{
			continue;
		}

		const Point& destp0 = pointsSrc[triangles[trIdx].points[0]];
		const Point& destp1 = pointsSrc[triangles[trIdx].points[1]];
		const Point& destp2 = pointsSrc[triangles[trIdx].points[2]];

		Point destp;
		destp.x = s * destp0.x + t * destp1.x + (1 - s - t) * destp2.x;
		destp.y = s * destp0.y + t * destp1.y + (1 - s - t) * destp2.y;

		destp.x = destp.x * ratio + p.x * (1 - ratio);
		destp.y = destp.y * ratio + p.y * (1 - ratio);

		return destp;
	}
}

__global__ 
void morphKernel(DeviceMorph* d_instance, double ratio)
{
	Point p;
	p.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	p.y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (!(p.x >= 0 && p.x < d_instance->d_output->width() && p.y >= 0 && p.y < d_instance->d_output->height()))
	{
		return;
	}

	Point srcPoint = computePosition(p, d_instance->d_pointsSrc, d_instance->d_pointsDest, d_instance->d_triangles, d_instance->_trianglesSize, ratio);
	Point destPoint = computePosition(p, d_instance->d_pointsDest, d_instance->d_pointsSrc, d_instance->d_triangles, d_instance->_trianglesSize, 1 - ratio);

	uchar4 srcPixel = d_instance->d_imageSrcTexture->cubicTex2D(srcPoint.x, srcPoint.y);
	uchar4 destPixel = d_instance->d_imageDestTexture->cubicTex2D(destPoint.x, destPoint.y);

	d_instance->d_output->at(p.x, p.y, 0, 0) = srcPixel.x * (1 - ratio) + destPixel.x * ratio;
	d_instance->d_output->at(p.x, p.y, 0, 1) = srcPixel.y * (1 - ratio) + destPixel.y * ratio;
	d_instance->d_output->at(p.x, p.y, 0, 2) = srcPixel.z * (1 - ratio) + destPixel.z * ratio;
}

__global__
void warpKernel(DeviceMorph* d_instance, double ratio, int way)
{
	Point p;
	p.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	p.y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (!(p.x >= 0 && p.x < d_instance->d_output->width() && p.y >= 0 && p.y < d_instance->d_output->height()))
	{
		return;
	}

	if (way == 1)
	{
		Point srcPoint = computePosition(p, d_instance->d_pointsSrc, d_instance->d_pointsDest, d_instance->d_triangles, d_instance->_trianglesSize, ratio);
		uchar4 srcPixel = tex2D<uchar4>(d_instance->d_imageDestTexture->tex, srcPoint.x + 0.5f, srcPoint.y + 0.5f);
		d_instance->d_output->at(p.x, p.y, 0, 0) = srcPixel.x;
		d_instance->d_output->at(p.x, p.y, 0, 1) = srcPixel.y;
		d_instance->d_output->at(p.x, p.y, 0, 2) = srcPixel.z;
	}
	else if (way == 2)
	{
		Point destPoint = computePosition(p, d_instance->d_pointsDest, d_instance->d_pointsSrc, d_instance->d_triangles, d_instance->_trianglesSize, ratio);
		uchar4 destPixel = tex2D<uchar4>(d_instance->d_imageSrcTexture->tex, destPoint.x + 0.5f, destPoint.y + 0.5f);
		d_instance->d_output->at(p.x, p.y, 0, 0) = destPixel.x;
		d_instance->d_output->at(p.x, p.y, 0, 1) = destPixel.y;
		d_instance->d_output->at(p.x, p.y, 0, 2) = destPixel.z;
	}
}

std::vector<cimg_library::CImg<unsigned char>> DeviceMorph::computeMorph(const size_t threadsX, const size_t threadsY) const
{
	int size = _output->size();
	cimg_library::CImg<unsigned char> cImg(_output->width(), _output->height(), _output->depth(), _output->spectrum());
	std::vector<cimg_library::CImg<unsigned char>> frames;

	dim3 threadsPerBlock(threadsX, threadsY);
	dim3 numBlocks((_output->width() / threadsPerBlock.x) + 1, (_output->height() / threadsPerBlock.y) + 1);

	double step = 0.02;
	for (double r = step; r <= 1.0; r += step)
	{
		morphKernel<<< numBlocks, threadsPerBlock >>>(d_instance, r);
		cudaMemcpy(cImg._data, _output->data(), sizeof(unsigned char) * size, cudaMemcpyDeviceToHost);
		frames.push_back(cImg);
		printf("Done with frame step %.3f\n", r);
	}

	return frames;
}


cimg_library::CImg<unsigned char> DeviceMorph::computeWarp(double ratio, int way, const size_t threadsX, const size_t threadsY) const
{
	int size = _output->size();

	cimg_library::CImg<unsigned char> cImg(_output->width(), _output->height(), _output->depth(), _output->spectrum());
	dim3 threadsPerBlock(threadsX, threadsY);
	dim3 numBlocks((_output->width() / threadsPerBlock.x) + 1, (_output->height() / threadsPerBlock.y) + 1);

	warpKernel<<< numBlocks, threadsPerBlock >>>(d_instance, ratio, way);
	cudaMemcpy(cImg._data, _output->data(), sizeof(unsigned char) * size, cudaMemcpyDeviceToHost);

	return cImg;
}

