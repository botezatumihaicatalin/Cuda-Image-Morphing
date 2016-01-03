#ifndef _DELAUNAY_H_
#define _DELAUNAY_H_

#include <vector>

#include "Geometry.h"

struct IndexTriangle
{
	int points[3];
};

std::vector<IndexTriangle> boyerWatson(const std::vector<Point> & pts);

#endif
