#include "Image.cuh"

const unsigned char* Image::data() const
{
	return _data;
}

size_t Image::width() const
{
	return _width;
}

size_t Image::height() const
{
	return _height;
}

size_t Image::depth() const
{
	return _depth;
}

size_t Image::spectrum() const
{
	return _spectrum;
}

size_t Image::size() const
{
	return _width * _height * _depth * _spectrum;
}

long Image::offset(const int& x, const int& y, const int& z, const int& c) const
{
	return x + y * (long)_width + z * (long)(_width * _height) + c * (long)(_width * _height * _depth);
}

const unsigned char& Image::at(const int& x, const int& y, const int& z, const int& c) const
{
	return _data[offset(x, y, z, c)];
}

unsigned char& Image::at(const int& x, const int& y, const int& z, const int& c)
{
	return _data[offset(x, y, z, c)];
}

double Image::cubic_atXY(const double& fx, const double& fy, const int& z, const int& c) const
{
	const double nfx = fx < 0 ? 0 : (fx > _width - 1 ? _width - 1 : fx);
	const double nfy = fy < 0 ? 0 : (fy > _height - 1 ? _height - 1 : fy);
	const int x = (int)nfx, y = (int)nfy;
	const double dx = nfx - x, dy = nfy - y;
	const int px = x - 1 < 0 ? 0 : x - 1, nx = dx > 0 ? x + 1 : x, ax = x + 2 >= _width ? _width - 1 : x + 2;
	const int py = y - 1 < 0 ? 0 : y - 1, ny = dy > 0 ? y + 1 : y, ay = y + 2 >= _height ? _height - 1 : y + 2;
	const double
		Ipp = (double)at(px, py, z, c), Icp = (double)at(x, py, z, c), Inp = (double)at(nx, py, z, c),
		Iap = (double)at(ax, py, z, c),
		Ip = Icp + 0.5f * (dx * (-Ipp + Inp) + dx * dx * (2 * Ipp - 5 * Icp + 4 * Inp - Iap) + dx * dx * dx * (-Ipp + 3 * Icp - 3 * Inp + Iap)),
		Ipc = (double)at(px, y, z, c), Icc = (double)at(x, y, z, c), Inc = (double)at(nx, y, z, c),
		Iac = (double)at(ax, y, z, c),
		Ic = Icc + 0.5f * (dx * (-Ipc + Inc) + dx * dx * (2 * Ipc - 5 * Icc + 4 * Inc - Iac) + dx * dx * dx * (-Ipc + 3 * Icc - 3 * Inc + Iac)),
		Ipn = (double)at(px, ny, z, c), Icn = (double)at(x, ny, z, c), Inn = (double)at(nx, ny, z, c),
		Ian = (double)at(ax, ny, z, c),
		In = Icn + 0.5f * (dx * (-Ipn + Inn) + dx * dx * (2 * Ipn - 5 * Icn + 4 * Inn - Ian) + dx * dx * dx * (-Ipn + 3 * Icn - 3 * Inn + Ian)),
		Ipa = (double)at(px, ay, z, c), Ica = (double)at(x, ay, z, c), Ina = (double)at(nx, ay, z, c),
		Iaa = (double)at(ax, ay, z, c),
		Ia = Ica + 0.5f * (dx * (-Ipa + Ina) + dx * dx * (2 * Ipa - 5 * Ica + 4 * Ina - Iaa) + dx * dx * dx * (-Ipa + 3 * Ica - 3 * Ina + Iaa));
	return Ic + 0.5f * (dy * (-Ip + In) + dy * dy * (2 * Ip - 5 * Ic + 4 * In - Ia) + dy * dy * dy * (-Ip + 3 * Ic - 3 * In + Ia));
}

DeviceImage::DeviceImage(size_t width, size_t height, size_t depth, size_t spectrum)
{
	_width = width;
	_height = height;
	_depth = depth;
	_spectrum = spectrum;
	cudaMalloc(&_data, sizeof(unsigned char) * size());
}

DeviceImage::DeviceImage(const cimg_library::CImg<unsigned char>& image)
{
	_width = image.width();
	_height = image.height();
	_depth = image.depth();
	_spectrum = image.spectrum();
	cudaMalloc(&_data, sizeof(unsigned char) * image.size());
	cudaMemcpy(_data, image.data(), sizeof(unsigned char) * image.size(), cudaMemcpyHostToDevice);
}

DeviceImage::~DeviceImage()
{
	cudaFree(_data);
}

