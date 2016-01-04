#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "Delaunay.h"
#include "Geometry.h"

template<class T>
void fastRemove(std::vector<T> & arr, size_t index)
{
	size_t lastIndex = arr.size() - 1;
	std::swap(arr[lastIndex], arr[index]);
	arr.pop_back();
}

struct PairHash {
	inline size_t operator()(const std::pair<size_t,size_t> & x) const {
		return std::hash<size_t>()(x.first) ^ std::hash<size_t>()(x.second);
	}
};

std::vector<IndexTriangle> boyerWatson(const std::vector<Point> & pts)
{
	Point p1, p2, p3;

	p1.x = -100000;
	p1.y = -100000;

	p2.x = 100000;
	p2.y = -900000;

	p3.x = 0;
	p3.y = 40000;

	std::vector<Point> clone;
	clone.push_back(p1);
	clone.push_back(p2);
	clone.push_back(p3);

	std::vector<IndexTriangle> triangs;
	
	IndexTriangle newTriang;
	newTriang.points[0] = 0;
	newTriang.points[1] = 1;
	newTriang.points[2] = 2;

	triangs.push_back(newTriang);

	std::vector<size_t> badTri;
	std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> edgesMap;
	std::vector<std::pair<size_t, size_t>> polygon;

	for (size_t idx = 0; idx < pts.size(); idx++)
	{
		clone.push_back(pts[idx]);

		for (size_t triIndex = 0; triIndex < triangs.size(); triIndex++)
		{
			Circle circum = findCircumcircle(clone[triangs[triIndex].points[0]], clone[triangs[triIndex].points[1]], clone[triangs[triIndex].points[2]]);
			if (!isInside(circum, pts[idx]))
			{
				continue;
			}
			badTri.push_back(triIndex);
			for (size_t pIndex = 0; pIndex < 3; pIndex++)
			{
				size_t nextPIndex = (pIndex + 1) % 3;
				std::pair<size_t, size_t> edge(triangs[triIndex].points[pIndex], triangs[triIndex].points[nextPIndex]);
				if (edge.first > edge.second)
				{
					std::swap(edge.first, edge.second);
				}
				auto val = edgesMap.find(edge);
				if (val == edgesMap.end()) {
					edgesMap[edge] = 0;
				};
				edgesMap[edge] ++;
			}
		}

		for (size_t i = 0; i < badTri.size(); i++)
		{
			for (size_t pIndex = 0; pIndex < 3; pIndex++)
			{
				size_t nextPIndex = (pIndex + 1) % 3;
				std::pair<size_t, size_t> edge(triangs[badTri[i]].points[pIndex], triangs[badTri[i]].points[nextPIndex]);
				if (edge.first > edge.second)
				{
					std::swap(edge.first, edge.second);
				}
				if (edgesMap[edge] == 1)
				{
					polygon.push_back(edge);
				}
			}
		}

		std::sort(badTri.begin(), badTri.end());
		while(!badTri.empty())
		{
			size_t& triangleIndex = badTri.back();
			fastRemove(triangs, triangleIndex);
			badTri.pop_back();
		}

		while(!polygon.empty())
		{
			std::pair<size_t, size_t> &edge = polygon.back();
			polygon.pop_back();
			
			IndexTriangle newTriang;
			newTriang.points[0] = edge.first;
			newTriang.points[1] = edge.second;
			newTriang.points[2] = idx + 3;
			
			triangs.push_back(newTriang);
		}

		edgesMap.clear();
	}

	for (size_t triIdx = 0; triIdx < triangs.size(); triIdx ++)
	{
		for (size_t pIndex = 0; pIndex < 3; pIndex++) {
			if (triangs[triIdx].points[pIndex] <= 2)
			{
				fastRemove(triangs, triIdx);
				triIdx --;
				break;
			}
			else 
			{
				triangs[triIdx].points[pIndex] -= 3;
			}
		}
	}

	return triangs;
}


//vector<int> sHull(vector<Point> & pts)
//{
//
//	vector<int> tri;
//	deque<int> hull;
//
//	int x0 = 0;
//	auto cmp = [&x0, &pts](Point & p1, Point & p2) -> bool { 
//		return dist(p1, pts[x0]) < dist(p2, pts[x0]); 
//	};
//
//	sort(pts.begin(), pts.end(), cmp);
//
//	int xj = 1;
//
//	Circle c = findCircumcircle(pts[x0], pts[xj], pts[2]);
//	int xk = 2;
//	for (int idx = 3; idx < pts.size(); idx++)
//	{
//		Circle circum = findCircumcircle(pts[x0], pts[xj], pts[idx]);
//		if (circum.radius < c.radius)
//		{
//			c = circum;
//			xk = idx;
//		}
//	}
//
//	if (cross(pts[x0], pts[xj], pts[xk]) < 0)
//	{
//		hull.push_back(x0);
//		hull.push_back(xj);
//		hull.push_back(xk);
//	}
//	else
//	{
//		hull.push_back(x0);
//		hull.push_back(xk);
//		hull.push_back(xj);
//	}
//
//	tri.push_back(x0);
//	tri.push_back(xj);
//	tri.push_back(xk);
//	
//	auto new_cmp = [&c](Point & p1, Point & p2) -> bool { 
//		return dist(p1, c.center) < dist(p2, c.center); 
//	};
//
//	sort(pts.begin(), pts.end(), new_cmp);
//
//	for (int idx = 3; idx < pts.size(); idx++)
//	{
//		while (hull.size() > 1)
//		{
//			if (cross(pts[idx], pts[hull[0]], pts[hull[1]]) < 0)
//			{
//				break;
//			}
//			tri.push_back(idx);
//			tri.push_back(hull[0]);
//			tri.push_back(hull[1]);
//			hull.pop_front();
//		}
//
//		while (hull.size() > 1)
//		{
//			int last = hull.size() - 1;
//			if (cross(pts[idx], pts[hull[last]], pts[hull[last - 1]]) > 0)
//			{
//				break;
//			}
//			tri.push_back(idx);
//			tri.push_back(hull[0]);
//			tri.push_back(hull[1]);
//			hull.pop_back();
//		}
//	}
//}