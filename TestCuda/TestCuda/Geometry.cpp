#include "Geometry.h"
#include <math.h>

double dist(const Point & p1, const Point & p2)
{
	return sqrt(sqr(p1.x - p2.x) + sqr(p1.y - p2.y));
}

Circle findCircumcircle(const Point & p1, const Point & p2, const Point & p3)
{
	Point mid1;
	mid1.x = (p1.x + p2.x) / 2;
	mid1.y = (p1.y + p2.y) / 2;

	Point mid2;
	mid2.x = (p2.x + p3.x) / 2;
	mid2.y = (p2.y + p3.y) / 2;

	double p12_dx = p1.x - p2.x;
	double p12_dy = p1.y - p2.y;

	double p23_dx = p2.x - p3.x;
	double p23_dy = p2.y - p3.y;

	if ((p12_dx == 0 && p23_dy == 0) || (p12_dy == 0 && p23_dx == 0))
	{
		Circle c;
		c.center.x = (p1.x + p3.x) / 2;
		c.center.y = (p1.y + p3.y) / 2;
		c.radius = dist(p1, c.center);
		return c;
	}
	
	if (p12_dx == 0 && p12_dy != 0 && p23_dy != 0 && p23_dx != 0)
	{
		double slope2 = (p2.y - p3.y) / (p2.x - p3.x);
		double bisect_slope2 = -1 / slope2;

		double y = (p1.y + p2.y) / 2;
		double x = (y - mid2.y + bisect_slope2 * mid2.x) / bisect_slope2;

		// (y - mid2.y) = bisect_slope2 * (x - mid2.x)
		// y = (p1.y + p2.y) / 2;
		// ((p1.y + p2.y) / 2 - mid2.y) = bisect_slope2 * (x - mid2.x);
		// x = ((p1.y + p2.y) / 2 - mid2.y + bisect_slope2 * mid2.x) / bisect_slope2;

		Circle c;
		c.center.x = x;
		c.center.y = y;
		c.radius = dist(p1, c.center);

		return c;
	}

	if (p12_dy == 0 && p12_dx != 0 && p23_dx != 0 && p23_dy != 0)
	{
		double slope2 = (p2.y - p3.y) / (p2.x - p3.x);
		double bisect_slope2 = -1 / slope2;

		double x = (p1.x + p2.x) / 2;
		double y = bisect_slope2 * (x - mid2.x) + mid2.y;
		
		Circle c;
		c.center.x = x;
		c.center.y = y;
		c.radius = dist(p1, c.center);

		return c;
		
		// x = (p1.x + p2.x) / 2;
		// (y - mid2.y) = bisect_slope2 * (x - mid2.x)
		// y = bisect_slope2 * ((p1.x + p2.x) / 2 - mid2.x) + mid2.y;
	}

	if (p12_dx != 0 && p12_dy != 0 && p23_dy == 0 && p23_dx != 0)
	{
		double slope1 = (p1.y - p2.y) / (p1.x - p2.x);
		double bisect_slope1 = -1 / slope1;

		double x = (p2.x + p3.x) / 2;
		double y = bisect_slope1 * (x - mid1.x) + mid1.y;

		// x = (p2.x + p3.x) / 2
		// (y - mid1.y) = bisect_slope1 * (x - mid1.x)
		// y = bisect_slope1 * (x - mid1.x) + mid1.y

		Circle c;
		c.center.x = x;
		c.center.y = y;
		c.radius = dist(p1, c.center);

		return c;
	}

	if (p12_dx != 0 && p12_dy != 0 && p23_dy != 0 && p23_dx == 0)
	{
		double slope1 = (p1.y - p2.y) / (p1.x - p2.x);
		double bisect_slope1 = -1 / slope1;

		double y = (p2.y + p3.y) / 2;
		double x = (y - mid1.y + mid1.x * bisect_slope1) / bisect_slope1;

		// (y - mid1.y) = bisect_slope1 * (x - mid1.x)
		// y = (p2.y + p3.y) / 2;
		// x = (y - mid1.y + mid1.x * bisect_slope1) / bisect_slope1;

		Circle c;
		c.center.x = x;
		c.center.y = y;
		c.radius = dist(p1, c.center);

		return c;
	}
	

	double slope1 = (p1.y - p2.y) / (p1.x - p2.x);
	double slope2 = (p2.y - p3.y) / (p2.x - p3.x);

	if (slope1 == slope2)
	{


		/*Circle c;
		c.center.x = x;
		c.center.y = y;
		c.radius = dist(p1, c.center);

		return c;*/
	}

	double bisect_slope1 = -1 / slope1;
	double bisect_slope2 = -1 / slope2;
	
	// (y - mid1.y) = bisect_slope1 * (x - mid1.x)
	// (y - mid2.y) = bisect_slope2 * (x - mid2.x)
	// y = bisect_slope2 * (x - mid2.x) + mid2.y
	// (bisect_slope2 * (x - mid2.x) + mid2.y - mid1.y) = bisect_slope1 * (x - mid1.x)
	// x * (bisect_slope2 - bisect_slope1) = bisect_slope2 * mid2.x - bisect_slope1 * mid1.x + mid1.y - mid2.y
	// x = (bisect_slope2 * mid2.x - bisect_slope1 * mid1.x + mid1.y - mid2.y) / (bisect_slope2 - bisect_slope1)

	double x = (bisect_slope2 * mid2.x - bisect_slope1 * mid1.x + mid1.y - mid2.y) / (bisect_slope2 - bisect_slope1);
	double y = (bisect_slope2 * (x - mid2.x) + mid2.y);

	Circle c;
	c.center.x = x;
	c.center.y = y;
	c.radius = dist(p1, c.center);

	return c;
}

bool isInside(const Circle & c, const Point & p)
{
	return dist(c.center, p) <= c.radius;
}

double cross(const Point &A, const Point &O, const Point &B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}