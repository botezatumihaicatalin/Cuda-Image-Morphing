#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <time.h>
#include <CImg.h>
#include <math.h>
#include <vector>
#include <string>

#include "Geometry.h"
#include "Delaunay.h"

#include <windows.h>

void drawPoints(cimg_library::CImg<unsigned char> & img, const std::vector<Point> & points)
{
	const unsigned char red[] = { 255,0,0 };
	for (int i = 0; i < points.size(); i++)
	{
		img.draw_circle(points[i].x, points[i].y, 3, red);
	}
}

void drawTriangulation(cimg_library::CImg<unsigned char> & img, const std::vector<Point> & points, const std::vector<IndexTriangle> & triangles)
{
	const unsigned char green[] = { 0,255,0 };
	for (int i = 0; i < triangles.size(); i++)
	{
		for (int pIndex = 0; pIndex < 3; pIndex ++)
		{
			int nextPIndex = (pIndex + 1) % 3;
			img.draw_line(
				points[triangles[i].points[pIndex]].x, 
				points[triangles[i].points[pIndex]].y, 
				points[triangles[i].points[nextPIndex]].x, 
				points[triangles[i].points[nextPIndex]].y, 
			green);
		}
	}
}

struct WarpInput
{
	const unsigned char * imageData;
	unsigned char * resultData;
	int width, height, depth, spectrum;

	const Point * pointsSrc, * pointsDest;
	int pointsSrcSize, pointsDestSize;

	const IndexTriangle * triangles;
	int trianglesSize;
};

void cudaFreeWarpInput(WarpInput * deviceInput) {
	WarpInput * hostInput = new WarpInput();

	cudaMemcpy(hostInput, deviceInput, sizeof(WarpInput), cudaMemcpyHostToDevice);
	
	cudaFree((void*)hostInput->imageData);
	cudaFree((void*)hostInput->resultData);
	cudaFree((void*)hostInput->pointsSrc);
	cudaFree((void*)hostInput->pointsDest);
	cudaFree((void*)hostInput->triangles);
	cudaFree((void*)deviceInput);

	delete hostInput;
}

WarpInput * cudaCreateWarpInput(cimg_library::CImg<unsigned char> & img, 
	const std::vector<Point> & pointsSrc,  
	const std::vector<Point> & pointsDest, const std::vector<IndexTriangle> & triangles)
{
	WarpInput * input = new WarpInput();

	input->width = img.width();
	input->height = img.height();
	input->spectrum = img.spectrum();
	input->depth = img.depth();

	const unsigned char * imgData = img.data();
	unsigned char * dImgData;
	cudaMalloc(&dImgData, sizeof(unsigned char) * img.size());
	cudaMemcpy(dImgData, imgData, sizeof(unsigned char) * img.size(), cudaMemcpyHostToDevice);
	input->imageData = dImgData;

	unsigned char * dResultData;
	cudaMalloc(&dResultData, sizeof(unsigned char) * img.size());
	cudaMemset(dResultData, 0, sizeof(unsigned char) * img.size());
	input->resultData = dResultData;

	const Point * pointsSrcData = pointsSrc.data();
	Point * dPointsSrcData;

	cudaMalloc(&dPointsSrcData, sizeof(Point) * pointsSrc.size());
	cudaMemcpy(dPointsSrcData, pointsSrcData, sizeof(Point) * pointsSrc.size(), cudaMemcpyHostToDevice);
	input->pointsSrc = dPointsSrcData;
	input->pointsSrcSize = pointsSrc.size();
	
	const Point * pointsDestData = pointsDest.data();
	Point * dPointsDestData;

	cudaMalloc(&dPointsDestData, sizeof(Point) * pointsDest.size());
	cudaMemcpy(dPointsDestData, pointsDestData, sizeof(Point) * pointsDest.size(), cudaMemcpyHostToDevice);
	input->pointsDest = dPointsDestData;
	input->pointsDestSize = pointsDest.size();

	const IndexTriangle * trianglesData = triangles.data();
	IndexTriangle * dTrianglesData;

	cudaMalloc(&dTrianglesData, sizeof(IndexTriangle) * triangles.size());
	cudaMemcpy(dTrianglesData, trianglesData, sizeof(IndexTriangle) * triangles.size(), cudaMemcpyHostToDevice);
	input->triangles = dTrianglesData;
	input->trianglesSize = triangles.size();

	WarpInput * dInput;
	cudaMalloc(&dInput, sizeof(WarpInput));
	cudaMemcpy(dInput, input, sizeof(WarpInput), cudaMemcpyHostToDevice);

	delete input;

	return dInput;
}

__host__ __device__ long at(const int & x, const int & y, const int & z, const int & c, const int & width, const int & height, const int & depth) 
{
	return x + y*(long)width + z*(long)(width*height) + c*(long)(width*height*depth);
}

__host__ __device__ double cubic_atXY(const double & fx, const double & fy, const int & z, const int & c, const int & width, const int & height, const int & depth, const unsigned char * img) {
      const double
        nfx = fx<0?0:(fx>width - 1?width - 1:fx),
        nfy = fy<0?0:(fy>height - 1?height - 1:fy);
      const int x = (int)nfx, y = (int)nfy;
      const double dx = nfx - x, dy = nfy - y;
      const int
        px = x - 1<0?0:x - 1, nx = dx>0?x + 1:x, ax = x + 2>=width?width - 1:x + 2,
        py = y - 1<0?0:y - 1, ny = dy>0?y + 1:y, ay = y + 2>=height?height - 1:y + 2;
      const double
        Ipp = (double)img[at(px,py,z,c,width,height,depth)], Icp = (double)img[at(x,py,z,c,width,height,depth)], Inp = (double)img[at(nx,py,z,c,width,height,depth)],
        Iap = (double)img[at(ax,py,z,c,width,height,depth)],
        Ip = Icp + 0.5f*(dx*(-Ipp + Inp) + dx*dx*(2*Ipp - 5*Icp + 4*Inp - Iap) + dx*dx*dx*(-Ipp + 3*Icp - 3*Inp + Iap)),
        Ipc = (double)img[at(px,y,z,c,width,height,depth)],  Icc = (double)img[at(x,y,z,c,width,height,depth)], Inc = (double)img[at(nx,y,z,c,width,height,depth)],
        Iac = (double)img[at(ax,y,z,c,width,height,depth)],
        Ic = Icc + 0.5f*(dx*(-Ipc + Inc) + dx*dx*(2*Ipc - 5*Icc + 4*Inc - Iac) + dx*dx*dx*(-Ipc + 3*Icc - 3*Inc + Iac)),
        Ipn = (double)img[at(px,ny,z,c,width,height,depth)], Icn = (double)img[at(x,ny,z,c,width,height,depth)], Inn = (double)img[at(nx,ny,z,c,width,height,depth)],
        Ian = (double)img[at(ax,ny,z,c,width,height,depth)],
        In = Icn + 0.5f*(dx*(-Ipn + Inn) + dx*dx*(2*Ipn - 5*Icn + 4*Inn - Ian) + dx*dx*dx*(-Ipn + 3*Icn - 3*Inn + Ian)),
        Ipa = (double)img[at(px,ay,z,c,width,height,depth)], Ica = (double)img[at(x,ay,z,c,width,height,depth)], Ina = (double)img[at(nx,ay,z,c,width,height,depth)],
        Iaa = (double)img[at(ax,ay,z,c,width,height,depth)],
        Ia = Ica + 0.5f*(dx*(-Ipa + Ina) + dx*dx*(2*Ipa - 5*Ica + 4*Ina - Iaa) + dx*dx*dx*(-Ipa + 3*Ica - 3*Ina + Iaa));
      return Ic + 0.5f*(dy*(-Ip + In) + dy*dy*(2*Ip - 5*Ic + 4*In - Ia) + dy*dy*dy*(-Ip + 3*Ic - 3*In + Ia));
}

__host__ __device__ void processPixel(const double & x, const double & y, WarpInput * input, const double & ratio = 1)
{
	if (!(x < input->width && x >= 0 && y < input->height && y >= 0))
	{
		return;
	}

	Point p;
	p.x = x;
	p.y = y;
	for (int trIdx = 0; trIdx < input->trianglesSize; trIdx++)
	{
		const Point & p1 = input->pointsDest[input->triangles[trIdx].points[0]];
		const Point & p2 = input->pointsDest[input->triangles[trIdx].points[1]];
		const Point & p3 = input->pointsDest[input->triangles[trIdx].points[2]];

		double bot = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
		double sTop = (p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y);
		double tTop = (p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y);

		double s = sTop / bot;
		double t = tTop / bot;

		if (!(s >= 0 && s <= 1 && t >= 0 && t <= 1 && (s + t) <= 1))
		{
			continue;
		}

		const Point & destp0 = input->pointsSrc[input->triangles[trIdx].points[0]];
		const Point & destp1 = input->pointsSrc[input->triangles[trIdx].points[1]];
		const Point & destp2 = input->pointsSrc[input->triangles[trIdx].points[2]];

		Point destp;
		destp.x = s * destp0.x + t * destp1.x + (1 - s - t) * destp2.x;
		destp.y = s * destp0.y + t * destp1.y + (1 - s - t) * destp2.y;

		destp.x = destp.x * ratio + p.x * (1 - ratio); 
		destp.y = destp.y * ratio + p.y * (1 - ratio); 

		for (int c = 0; c < input->spectrum; c++)
		{
			long offsetResult = at(x, y, 0, c, input->width, input->height, input->depth);
			input->resultData[offsetResult] = cubic_atXY(destp.x, destp.y, 0, c, input->width, input->height, input->depth, input->imageData);
		}

		break;
	}
}

__global__ void morphKernel(WarpInput * inputSrc, WarpInput * inputDest, double ratio = 1)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	processPixel(x, y, inputSrc, ratio);
	processPixel(x, y, inputDest, 1 - ratio);

	for (int c = 0; c < 3; c++)
	{
		long offset = at(x, y, 0, c, inputSrc->width, inputSrc->height, inputSrc->depth); 
		inputSrc->resultData[offset] = (1.0 - ratio) * inputSrc->resultData[offset] + ratio * inputDest->resultData[offset]; 
	}
}


cimg_library::CImg<unsigned char> warp(cimg_library::CImg<unsigned char> & img, 
	const std::vector<Point> & pointsSrc,  
	const std::vector<Point> & pointsDest, const std::vector<IndexTriangle> & triangles,
	const double & ratio = 1)
{
	cimg_library::CImg<unsigned char> result(img);
	result.fill(1);
	int width = img.width();
	int height = img.height();

	WarpInput warpInput;
	warpInput.imageData = img.data();
	warpInput.resultData = result.data();
	warpInput.width = img.width();
	warpInput.height = img.height();
	warpInput.depth = img.depth();
	warpInput.spectrum = img.spectrum();
	warpInput.pointsSrc = pointsSrc.data();
	warpInput.pointsSrcSize = pointsSrc.size();
	warpInput.pointsDest = pointsDest.data();
	warpInput.pointsDestSize = pointsDest.size();
	warpInput.triangles = triangles.data();
	warpInput.trianglesSize = triangles.size();

	cimg_forXY(img,x,y)
	{
		processPixel(x, y, &warpInput, ratio);
	}

	return result;
}


int main()
{
	cimg_library::CImg<unsigned char> imageSrc("test1/img2.jpg");
	cimg_library::CImg<unsigned char> imageDest("test1/catface.jpg");

	if (!(imageSrc.width() == imageDest.width() && 
		imageSrc.height() == imageDest.height() && 
		imageSrc.spectrum() == imageDest.spectrum()))
	{
		return 1;
	}

	cimg_library::CImg<unsigned char> outSrc(imageSrc);
	cimg_library::CImg<unsigned char> outDest(imageDest);

	int * t;
	cudaMalloc(&t, sizeof(int) * 1000);
	cudaFree(t);

	std::vector<Point> pointsSrc(4);
	pointsSrc[0].x = 0;
	pointsSrc[0].y = 0;

	pointsSrc[1].x = imageSrc.width() - 1;
	pointsSrc[1].y = imageSrc.height() - 1;

	pointsSrc[2].x = imageSrc.width() - 1;
	pointsSrc[2].y = 0;

	pointsSrc[3].x = 0;
	pointsSrc[3].y = imageSrc.height() - 1;

	std::vector<Point> pointsDest(pointsSrc);

	std::vector<IndexTriangle> triang = boyerWatson(pointsSrc);

	outSrc.assign(imageSrc);
	drawTriangulation(outSrc, pointsSrc, triang);
	drawPoints(outSrc, pointsSrc);

	outDest.assign(imageDest);
	drawTriangulation(outDest, pointsDest, triang);
	drawPoints(outDest, pointsDest);

	cimg_library::CImgDisplay drawSrc(outSrc, "Source morph");
	cimg_library::CImgDisplay drawDest(outDest, "Dest morph");

	while (!drawSrc.is_closed() && !drawDest.is_closed()) {
		cimg_library::CImgDisplay::wait(drawSrc, drawDest);
		if (drawSrc.button() && drawSrc.mouse_y() >= 0 && drawSrc.mouse_x() >= 0) {
			const int y = drawSrc.mouse_y();
			const int x = drawSrc.mouse_x();

			Point p;
			p.x = x;
			p.y = y;

			bool next = false;
			for (int i = 0; i < pointsSrc.size() && !next; i++) 
			{
				if (dist(p, pointsSrc[i]) < 10.0) 
				{
					pointsSrc[i] = p;
					next = true;
				}
			}

			if (!next) 
			{
				pointsSrc.push_back(p);
				pointsDest.push_back(p);
			}

			triang = boyerWatson(pointsSrc);
			
			outSrc.assign(imageSrc);
			
			drawTriangulation(outSrc, pointsSrc, triang);
			drawPoints(outSrc, pointsSrc);
			
			outSrc.display(drawSrc);

			outDest.assign(imageDest);
			
			drawTriangulation(outDest, pointsDest, triang);
			drawPoints(outDest, pointsDest);
			
			outDest.display(drawDest);
		}

		if (drawDest.button() && drawDest.mouse_y() >= 0 && drawDest.mouse_x() >= 0) {
			const int y = drawDest.mouse_y();
			const int x = drawDest.mouse_x();
			
			Point p;
			p.x = x;
			p.y = y;

			for (int i = 0; i < pointsDest.size(); i++) 
			{
				if (dist(p, pointsDest[i]) < 10.0) 
				{
					pointsDest[i] = p;
					break;
				}
			}
			
			outDest.assign(imageDest);
			
			drawTriangulation(outDest, pointsDest, triang);
			drawPoints(outDest, pointsDest);
			
			outDest.display(drawDest);
		}
	}

	drawSrc.close();
	drawDest.close();

	WarpInput * dWarpInputSrc = cudaCreateWarpInput(imageSrc, pointsSrc, pointsDest, triang);
	WarpInput * dWarpInputDest = cudaCreateWarpInput(imageDest, pointsDest, pointsSrc, triang);
	
	WarpInput * warpInputSrc = new WarpInput();
	WarpInput * warpInputDest = new WarpInput();

	dim3 threadsPerBlock(32, 32); 
	dim3 numBlocks((imageSrc.width() / threadsPerBlock.x) + 1, (imageSrc.height() / threadsPerBlock.y) + 1);

	cimg_library::CImgDisplay result(imageSrc, "Morphing animation");

	std::vector<cimg_library::CImg<unsigned char>> frames;
	
	int count = 0;
	double step = 0.02;
	for (double r = step; r <= 1.0; r += step) {
		count ++;
		
		morphKernel<<< numBlocks, threadsPerBlock >>>(dWarpInputSrc, dWarpInputDest, r);
		cudaMemcpy(warpInputSrc, dWarpInputSrc, sizeof(WarpInput), cudaMemcpyDeviceToHost);
		cudaMemcpy(imageSrc._data, warpInputSrc->resultData, sizeof(unsigned char) * imageSrc.size(), cudaMemcpyDeviceToHost);

		/*std::string fileName = "Test.jpg";
		fileName = std::to_string((long long)count) + fileName;
		fileName = "results/" + fileName;
		imageSrc.save(fileName.c_str());*/
		frames.push_back(imageSrc);
		printf("Done with frame step %.3f\n", r);
	}

	cudaFreeWarpInput(dWarpInputSrc);
	cudaFreeWarpInput(dWarpInputDest);

	double duration = 2000;
	double wait = duration / frames.size();

	int index = 0;
	int way = 1;
	while(!result.is_closed()) {
		frames[index].display(result);
		Sleep(wait);
		index += way;
		if (index >= count) {
			index = frames.size() - 1;
			way = -1;
		} else if (index <= -1) {
			index = 0;
			way = 1;
		}
	}

	return 0;
}