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

cimg_library::CImg<unsigned char> warp(cimg_library::CImg<unsigned char> & img, 
	const std::vector<Point> & pointsSrc,  
	const std::vector<Point> & pointsDest, const std::vector<IndexTriangle> & triangles)
{
	cimg_library::CImg<unsigned char> result(img);
	result.fill(0);
	int width = img.width();
	int height = img.height();
	cimg_forXY(img,x,y)
	{
		Point p;
		p.x = x;
		p.y = y;
		for (int trIdx = 0; trIdx < triangles.size(); trIdx++)
		{
			const Point & p1 = pointsSrc[triangles[trIdx].points[0]];
			const Point & p2 = pointsSrc[triangles[trIdx].points[1]];
			const Point & p3 = pointsSrc[triangles[trIdx].points[2]];

			double bot = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
			double sTop = (p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y);
			double tTop = (p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y);

			double s = sTop / bot;
			double t = tTop / bot;

			if (!(s >= 0 && s <= 1 && t >= 0 && t <= 1 && (s + t) <= 1))
			{
				continue;
			}

			const Point & dest_p0 = pointsDest[triangles[trIdx].points[0]];
			const Point & dest_p1 = pointsDest[triangles[trIdx].points[1]];
			const Point & dest_p2 = pointsDest[triangles[trIdx].points[2]];

			Point dest_p;
			dest_p.x = s * dest_p0.x + t * dest_p1.x + (1 - s - t) * dest_p2.x;
			dest_p.y = s * dest_p0.y + t * dest_p1.y + (1 - s - t) * dest_p2.y;

			for (int c = 0; c < 3; c++)
			{
				result((unsigned int)round(dest_p.x), (unsigned int)round(dest_p.y), 0, c) = img(x, y, 0, c);
			}

			printf("%.3f %.3f %.3f %.3f \n", dest_p.x, dest_p.y, p.x, p.y);

			break;
		}
	}

	return result;
}


int main()
{
	cimg_library::CImg<unsigned char> visu("lena2.jpg");
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