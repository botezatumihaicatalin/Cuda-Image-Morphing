#include <algorithm>
#include <unordered_map>

#include "Delaunay.h"
#include "Geometry.h"

template <class T>
void fastRemove(std::vector<T>& arr, size_t index)
{
	size_t size = arr.size();
	if (size == 0)
	{
		return;
	}
	size_t lastIndex = size - 1;
	std::swap(arr[lastIndex], arr[index]);
	arr.pop_back();
}

struct PairHash
{
	size_t operator()(const std::pair<size_t, size_t>& x) const
	{
		return std::hash<size_t>()(x.first) ^ std::hash<size_t>()(x.second);
	}
};

std::vector<IndexTriangle> boyerWatson(const std::vector<Point>& pts)
{
	Point p1, p2, p3;

	p1.x = -10000000;
	p1.y = -10000000;

	p2.x = 10000000;
	p2.y = -90000000;

	p3.x = 0;
	p3.y = 4000000;

	std::vector<Point> clone;
	clone.push_back(p1);
	clone.push_back(p2);
	clone.push_back(p3);

	std::vector<IndexTriangle> triangs;

	IndexTriangle firstTriang;
	firstTriang.points[0] = 0;
	firstTriang.points[1] = 1;
	firstTriang.points[2] = 2;

	triangs.push_back(firstTriang);

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
				if (val == edgesMap.end())
				{
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
		while (!badTri.empty())
		{
			size_t& triangleIndex = badTri.back();
			fastRemove(triangs, triangleIndex);
			badTri.pop_back();
		}

		while (!polygon.empty())
		{
			std::pair<size_t, size_t>& edge = polygon.back();
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
		for (size_t pIndex = 0; pIndex < 3; pIndex++)
		{
			if (triangs[triIdx].points[pIndex] <= 2)
			{
				fastRemove(triangs, triIdx);
				triIdx --;
				break;
			}
			triangs[triIdx].points[pIndex] -= 3;
		}
	}

	return triangs;
}


