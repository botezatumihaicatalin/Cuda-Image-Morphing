#include "Morph.cuh"
#include "Image.cuh"

__host__ __device__ void processPixel(const double & x, const double & y, Image * input, Image * output, const Point * pointsSrc, const Point * pointsDest, const IndexTriangle * triangles, const int & trianglesSize, const double & ratio = 1)
{
	Point p;
	p.x = x;
	p.y = y;
	for (int trIdx = 0; trIdx < trianglesSize; trIdx++)
	{
		const Point & p1 = pointsDest[triangles[trIdx].points[0]];
		const Point & p2 = pointsDest[triangles[trIdx].points[1]];
		const Point & p3 = pointsDest[triangles[trIdx].points[2]];

		double bot = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
		double sTop = (p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y);
		double tTop = (p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y);

		double s = sTop / bot;
		double t = tTop / bot;

		if (!(s >= 0 && s <= 1 && t >= 0 && t <= 1 && (s + t) <= 1))
		{
			continue;
		}

		const Point & destp0 = pointsSrc[triangles[trIdx].points[0]];
		const Point & destp1 = pointsSrc[triangles[trIdx].points[1]];
		const Point & destp2 = pointsSrc[triangles[trIdx].points[2]];

		Point destp;
		destp.x = s * destp0.x + t * destp1.x + (1 - s - t) * destp2.x;
		destp.y = s * destp0.y + t * destp1.y + (1 - s - t) * destp2.y;

		destp.x = destp.x * ratio + p.x * (1 - ratio); 
		destp.y = destp.y * ratio + p.y * (1 - ratio); 

		for (int c = 0; c < input->spectrum; c++)
		{
			output->at(x, y, 0, c) = input->cubic_atXY(destp.x, destp.y, 0, c);
		}

		break;
	}
}

__global__ void morphKernel(Image * inputSrc, Image * outputSrc, Image * inputDest, Image * outputDest, const Point * pointsSrc, const Point * pointsDest, const IndexTriangle * triangles, int trianglesSize, double ratio = 1)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (!(x < inputSrc->width && x >= 0 && y < inputSrc->height && y >= 0))
	{
		return;
	}

	processPixel(x, y, inputSrc, outputSrc, pointsSrc, pointsDest, triangles, trianglesSize, ratio);
	processPixel(x, y, inputDest, outputDest, pointsDest, pointsSrc, triangles, trianglesSize, 1 - ratio);

	for (int c = 0; c < 3; c++)
	{
		outputSrc->at(x, y, 0, c) = (1.0 - ratio) * outputSrc->at(x, y, 0, c) + ratio * outputDest->at(x, y, 0, c);
	}
}

__host__ std::vector<cimg_library::CImg<unsigned char>> morph(const cimg_library::CImg<unsigned char> & imageSrc, const cimg_library::CImg<unsigned char> & imageDest, const std::vector<Point> & pointsSrc, const std::vector<Point> & pointsDest, const std::vector<IndexTriangle> & triangles)
{
	Image * dImgSrc = deviceImageFromCImg(imageSrc);
	Image * dImgDest = deviceImageFromCImg(imageDest);

	Image * dImgSrcOut = deviceImageFromCImg(imageSrc);
	Image * dImgDestOut = deviceImageFromCImg(imageDest);

	const Point * pointsSrcData = pointsSrc.data();
	Point * dPointsSrcData;
	cudaMalloc(&dPointsSrcData, sizeof(Point) * pointsSrc.size());
	cudaMemcpy(dPointsSrcData, pointsSrcData, sizeof(Point) * pointsSrc.size(), cudaMemcpyHostToDevice);
	
	const Point * pointsDestData = pointsDest.data();
	Point * dPointsDestData;
	cudaMalloc(&dPointsDestData, sizeof(Point) * pointsDest.size());
	cudaMemcpy(dPointsDestData, pointsDestData, sizeof(Point) * pointsDest.size(), cudaMemcpyHostToDevice);

	const IndexTriangle * trianglesData = triangles.data();
	IndexTriangle * dTrianglesData;
	cudaMalloc(&dTrianglesData, sizeof(IndexTriangle) * triangles.size());
	cudaMemcpy(dTrianglesData, trianglesData, sizeof(IndexTriangle) * triangles.size(), cudaMemcpyHostToDevice);

	int trianglesSize = triangles.size();

	dim3 threadsPerBlock(16, 16); 
	dim3 numBlocks((imageSrc.width() / threadsPerBlock.x) + 1, (imageSrc.height() / threadsPerBlock.y) + 1);

	cimg_library::CImg<unsigned char> cImg(imageSrc);
	Image * img = new Image();

	std::vector<cimg_library::CImg<unsigned char>> frames;
	
	double step = 0.02;
	for (double r = step; r <= 1.0; r += step) {
		morphKernel<<< numBlocks, threadsPerBlock >>>(dImgSrc, dImgSrcOut, dImgDest, dImgDestOut, dPointsSrcData, dPointsDestData, dTrianglesData, trianglesSize, r);
		cudaMemcpy(img, dImgSrcOut, sizeof(Image), cudaMemcpyDeviceToHost);
		cudaMemcpy(cImg._data, img->data, sizeof(unsigned char) * cImg.size(), cudaMemcpyDeviceToHost);
		frames.push_back(cImg);
		printf("Done with frame step %.3f\n", r);
	}

	return frames;
}