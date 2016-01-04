#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <time.h>
#include <CImg.h>
#include <math.h>
#include <vector>

#include "Geometry.h"
#include "Delaunay.h"

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
	const unsigned char * img;
	unsigned char * result;
	int width, height, depth, spectrum;

	const Point * pointsSrc, * pointsDest;
	int pointsSrcSize, pointsDestSize;

	const IndexTriangle * triangles;
	int trianglesSize;
};

void processPixel(int x, int y, WarpInput * input)
{
	Point p;
	p.x = x;
	p.y = y;
	for (int trIdx = 0; trIdx < input->trianglesSize; trIdx++)
	{
		const Point & p1 = input->pointsSrc[input->triangles[trIdx].points[0]];
		const Point & p2 = input->pointsSrc[input->triangles[trIdx].points[1]];
		const Point & p3 = input->pointsSrc[input->triangles[trIdx].points[2]];

		double bot = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
		double sTop = (p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y);
		double tTop = (p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y);

		double s = sTop / bot;
		double t = tTop / bot;

		if (!(s >= 0 && s <= 1 && t >= 0 && t <= 1 && (s + t) <= 1))
		{
			continue;
		}

		const Point & destp0 = input->pointsDest[input->triangles[trIdx].points[0]];
		const Point & destp1 = input->pointsDest[input->triangles[trIdx].points[1]];
		const Point & destp2 = input->pointsDest[input->triangles[trIdx].points[2]];

		Point destp;
		destp.x = s * destp0.x + t * destp1.x + (1 - s - t) * destp2.x;
		destp.y = s * destp0.y + t * destp1.y + (1 - s - t) * destp2.y;

		Point destpRounded;
		destpRounded.x = round(destp.x);
		destpRounded.y = round(destp.y);

		for (int c = 0; c < 3; c++)
		{
			long offsetNew = destpRounded.x + destpRounded.y*(long)input->width + c*(long)(input->width * input->height * input->depth);
			long offset = x + y*(long)input->width + c*(long)(input->width * input->height * input->depth);
			input->result[offsetNew] = input->img[offset];
		}

		break;
	}
}

cimg_library::CImg<unsigned char> warp(cimg_library::CImg<unsigned char> & img, 
	const std::vector<Point> & pointsSrc,  
	const std::vector<Point> & pointsDest, const std::vector<IndexTriangle> & triangles)
{
	cimg_library::CImg<unsigned char> result(img);
	result.fill(0);
	int width = img.width();
	int height = img.height();

	WarpInput warpInput;
	warpInput.img = img.data();
	warpInput.result = result.data();
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
		processPixel(x, y, &warpInput);
	}

	return result;
}


int main()
{
	cimg_library::CImg<unsigned char> visu("important2.jpg");
	cimg_library::CImg<unsigned char> out1(visu);
	cimg_library::CImg<unsigned char> out2(visu);

	std::vector<Point> pointsSrc(4);
	pointsSrc[0].x = 0;
	pointsSrc[0].y = 0;

	pointsSrc[1].x = visu.width() - 1;
	pointsSrc[1].y = visu.height() - 1;

	pointsSrc[2].x = visu.width() - 1;
	pointsSrc[2].y = 0;

	pointsSrc[3].x = 0;
	pointsSrc[3].y = visu.height() - 1;

	std::vector<Point> pointsDest(pointsSrc);

	std::vector<IndexTriangle> triang = boyerWatson(pointsSrc);

	out1.assign(visu);
	drawTriangulation(out1, pointsSrc, triang);
	drawPoints(out1, pointsSrc);

	out2.assign(visu);
	drawPoints(out2, pointsDest);

	cimg_library::CImgDisplay drawSource(out2,"Source morph");
	cimg_library::CImgDisplay drawDest(out2,"Dest morph");

	while (!drawSource.is_closed() && !drawDest.is_closed()) {
		cimg_library::CImgDisplay::wait(drawSource, drawDest);
		if (drawSource.button() && drawSource.mouse_y() >= 0 && drawSource.mouse_x() >= 0) {
			const int y = drawSource.mouse_y();
			const int x = drawSource.mouse_x();

			Point p;
			p.x = x;
			p.y = y;

			pointsSrc.push_back(p);

			triang = boyerWatson(pointsSrc);
			
			out1.assign(visu);
			
			drawTriangulation(out1, pointsSrc, triang);
			drawPoints(out1, pointsSrc);
			
			out1.display(drawSource);
		}

		if (drawDest.button() && drawDest.mouse_y() >= 0 && drawDest.mouse_x() >= 0) {
			const int y = drawDest.mouse_y();
			const int x = drawDest.mouse_x();

			Point p;
			p.x = x;
			p.y = y;

			pointsDest.push_back(p);
			
			out2.assign(visu);
			drawPoints(out2, pointsDest);
			out2.display(drawDest);
		}
	}

	drawSource.close();
	drawDest.close();

	cimg_library::CImg<unsigned int> res = warp(visu, pointsSrc, pointsDest, triang);

	res.display();

	return 0;
}