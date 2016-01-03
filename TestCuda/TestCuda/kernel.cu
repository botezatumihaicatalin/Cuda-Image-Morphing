#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <time.h>
#include <CImg.h>
#include <math.h>
#include <vector>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "Geometry.h"
#include "Delaunay.h"

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

int main()
{
	cimg_library::CImg<unsigned char> visu(500, 400, 1, 3, 0);

	std::vector<Point> points(6);
	points[0].x = 10;
	points[0].y = 10;

	points[1].x = 450;
	points[1].y = 350;

	points[2].x = 450;
	points[2].y = 10;

	points[3].x = 10;
	points[3].y = 350;

	points[4].x = 200;
	points[4].y = 200;

	points[5].x = 222;
	points[5].y = 222;

	std::vector<IndexTriangle> t = boyerWatson(points);

	visu.fill(100);
	drawTriangulation(visu, points, t);

	cimg_library::CImgDisplay draw_disp(visu,"Intensity profile");

	while (!draw_disp.is_closed()) {
		
	}


	return 0;
}