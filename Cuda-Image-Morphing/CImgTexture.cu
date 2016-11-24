#include "CImgTexture.cuh"

CImgTexture::CImgTexture(const cimg_library::CImg<unsigned char>& image)
{
	size_t imageSize = image.size();
	size_t imageWidth = image.width();
	size_t imageHeight = image.height();
	size_t widthHeight = image.width() * image.height();
	const unsigned char* imageData = image.data();

	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	cudaMallocArray(&cuArray, &channelFormat, imageWidth, imageHeight);

	unsigned char* newData = new unsigned char[widthHeight * 4];
	memset(newData, 255, widthHeight * 4 * sizeof(unsigned char));
	for (size_t i = 0; i < imageSize; i ++)
	{
		size_t index = i / widthHeight;
		size_t offset = i % widthHeight;
		newData[4 * offset + index] = imageData[i];
	}
	cudaMemcpyToArray(cuArray, 0, 0, newData, widthHeight * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	/*texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = 0;*/

	cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}

CImgTexture::~CImgTexture()
{
	cudaDestroyTextureObject(tex);
	cudaFreeArray(cuArray);
}

__device__ uchar4 CImgTexture::linearTex2D(float x, float y)
{
	return tex2D<uchar4>(tex, x, y);
}

__host__ __device__ float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f / 3.0f - 0.5f * t * t * a;
	else if (t < 2.0f) return a * a * a / 6.0f;
	else return 0.0f;
}

__device__ uchar4 CImgTexture::cubicTex2D(float x, float y)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
	float2 index;
	index.x = floor(coord_grid.x);
	index.y = floor(coord_grid.y);

	float2 fraction;
	fraction.x = coord_grid.x - index.x;
	fraction.y = coord_grid.y - index.y;

	index.x += 0.5f; //move from [-0.5, extent-0.5] to [0, extent]
	index.y += 0.5f; //move from [-0.5, extent-0.5] to [0, extent]

	uchar4 result;
	memset(&result, 0, sizeof(result));
	for (float y = -1; y < 2.5f; y++)
	{
		float bsplineY = bspline(y - fraction.y);
		float v = index.y + y;
		for (float x = -1; x < 2.5f; x++)
		{
			float bsplineXY = bspline(x - fraction.x) * bsplineY;
			float u = index.x + x;
			uchar4 pixel = tex2D<uchar4>(tex, u, v);
			result.x += pixel.x * bsplineXY;
			result.y += pixel.y * bsplineXY;
			result.z += pixel.z * bsplineXY;
			result.w += pixel.w * bsplineXY;
		}
	}
	return result;
}

