#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <time.h>
#include <CImg.h>

using namespace cimg_library;

//long offset(const int x, const int y=0, const int z=0, const int c=0) const {
//    return x + y*(long)_width + z*(long)(_width*_height) + c*(long)(_width*_height*_depth);
//}

__global__ void useClass(char * data1, char * data2, char * output, float ratio, int width, int height, int depth, int spectrum)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < width * height * depth * spectrum)
	{
		output[index] = data1[index] * (1 - ratio) + data2[index] * ratio;
	}
}

int main()
{
	CImg<char> * img = new CImg<char>("important.jpg");
	CImg<char> * img2 = new CImg<char>("important2.jpg");

	int width = img->width();
	int height = img->height();
	int depth = img->depth();
	int spectrum = img->spectrum();
	int size = img->size();

	CImg<char> * output = new CImg<char>(width, height, depth, spectrum);

	char * d_imgData1;
	char * d_imgData2;
	char * d_imgOutput;

	cudaMalloc(&d_imgData1, sizeof(char) * size);
	cudaMalloc(&d_imgData2, sizeof(char) * size);
	cudaMalloc(&d_imgOutput, sizeof(char) * size);
	
	cudaMemcpy(d_imgData1, img->data(), sizeof(char) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_imgData2, img2->data(), sizeof(char) * size, cudaMemcpyHostToDevice);

	int numThreads = 512;
	int numBlocks = (size / 512) + 1;

	useClass<<< numBlocks, numThreads >>>(d_imgData1, d_imgData2, d_imgOutput, 1, width, height, depth, spectrum);
	cudaDeviceSynchronize(); 

	cudaMemcpy(output->_data, d_imgOutput, sizeof(char) * size, cudaMemcpyDeviceToHost);

	output->save("output.jpg");

	return 0;
}