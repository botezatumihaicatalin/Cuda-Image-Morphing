#ifndef _MORPH_H_
#define _MORPH_H_

#include <vector>
#include <CImg.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "Delaunay.h"

__host__ std::vector<cimg_library::CImg<unsigned char>> morph(const cimg_library::CImg<unsigned char> & imageSrc, 
	const cimg_library::CImg<unsigned char> & imageDest,
	const std::vector<Point> & pointsSrc, const std::vector<Point> & pointsDest, 
	const std::vector<IndexTriangle> & triangles);

#endif