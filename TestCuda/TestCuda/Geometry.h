#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#define sqr(x) ((x)*(x))

struct Point
{
	double x;
	double y;
};

struct Triangle
{
	Point points[3];
};

struct Circle
{
	Point center;
	double radius;
};

double dist(const Point& p1, const Point& p2);
Circle findCircumcircle(const Point& p1, const Point& p2, const Point& p3);
bool isInside(const Circle& c, const Point& p);
double cross(const Point& A, const Point& O, const Point& B);

#endif

