#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <time.h>
#include <CImg.h>
#include <vector>

#include "Geometry.h"
#include "Delaunay.h"

#include "Morph.cuh"

#include <windows.h>

void drawPoints(cimg_library::CImg<unsigned char> & img, const std::vector<Point> & points)
{
	const unsigned char red[] = { 255,0,0 };
	for (size_t i = 0; i < points.size(); i++)
	{
		img.draw_circle(points[i].x, points[i].y, 3, red);

	}
}

void drawTriangulation(cimg_library::CImg<unsigned char> & img, const std::vector<Point> & points, const std::vector<IndexTriangle> & triangles)
{
	const unsigned char green[] = { 0,255,0 };
	for (size_t i = 0; i < triangles.size(); i++)
	{
		for (size_t pIndex = 0; pIndex < 3; pIndex ++)
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

int main()
{
	cimg_library::CImg<unsigned char> imageSrc("test1/img2.jpg");
	cimg_library::CImg<unsigned char> imageDest("test1/img1.jpg");

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

	for (int i = 0; i < 0; i ++)
	{
		Point p;
		p.x = rand() % (imageSrc.width() - 1) + 1;
		p.y = rand() % (imageSrc.height() - 1) + 1;
		pointsSrc.push_back(p);
		pointsDest.push_back(p);
	}

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
			for (size_t i = 0; i < pointsSrc.size() && !next; i++) 
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

			for (size_t i = 0; i < pointsDest.size(); i++) 
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
	};

	cimg_library::CImgDisplay result(imageSrc, "Morphing animation");
	
	clock_t tStart = clock();
	DeviceMorph dMorph(imageSrc, imageDest, pointsSrc, pointsDest, triang);
	std::vector<cimg_library::CImg<unsigned char>> frames = dMorph.computeMorph();
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

	double duration = 2000;
	double wait = duration / frames.size();

	int index = 0;
	int way = 1;
	while(!result.is_closed()) {
		frames[index].display(result);
		Sleep(wait);
		index += way;
		if (index >= (int)frames.size()) {
			index = (int)frames.size() - 1;
			way = -1;
		} else if (index <= -1) {
			index = 0;
			way = 1;
		}
	}

	return 0;
}