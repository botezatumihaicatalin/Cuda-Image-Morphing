#include "Image.cuh"

long Image::offset(const int & x, const int & y, const int & z, const int & c) const
{
	return x + y*(long)width + z*(long)(width*height) + c*(long)(width*height*depth);
}
const unsigned char & Image::at(const int & x, const int & y, const int & z, const int & c) const
{
	return data[offset(x, y, z, c)];
}

unsigned char & Image::at(const int & x, const int & y, const int & z, const int & c)
{
	return data[offset(x, y, z, c)];
}

double Image::cubic_atXY(const double & fx, const double & fy, const int & z, const int & c) const {
	const double nfx = fx<0?0:(fx>width - 1?width - 1:fx);
	const double nfy = fy<0?0:(fy>height - 1?height - 1:fy);
	const int x = (int)nfx, y = (int)nfy;
	const double dx = nfx - x, dy = nfy - y;
	const int px = x - 1<0?0:x - 1, nx = dx>0?x + 1:x, ax = x + 2>=width?width - 1:x + 2;
	const int py = y - 1<0?0:y - 1, ny = dy>0?y + 1:y, ay = y + 2>=height?height - 1:y + 2;
	const double
	Ipp = (double)at(px,py,z,c), Icp = (double)at(x,py,z,c), Inp = (double)at(nx,py,z,c),
	Iap = (double)at(ax,py,z,c),
	Ip = Icp + 0.5f*(dx*(-Ipp + Inp) + dx*dx*(2*Ipp - 5*Icp + 4*Inp - Iap) + dx*dx*dx*(-Ipp + 3*Icp - 3*Inp + Iap)),
	Ipc = (double)at(px,y,z,c),  Icc = (double)at(x,y,z,c), Inc = (double)at(nx,y,z,c),
	Iac = (double)at(ax,y,z,c),
	Ic = Icc + 0.5f*(dx*(-Ipc + Inc) + dx*dx*(2*Ipc - 5*Icc + 4*Inc - Iac) + dx*dx*dx*(-Ipc + 3*Icc - 3*Inc + Iac)),
	Ipn = (double)at(px,ny,z,c), Icn = (double)at(x,ny,z,c), Inn = (double)at(nx,ny,z,c),
	Ian = (double)at(ax,ny,z,c),
	In = Icn + 0.5f*(dx*(-Ipn + Inn) + dx*dx*(2*Ipn - 5*Icn + 4*Inn - Ian) + dx*dx*dx*(-Ipn + 3*Icn - 3*Inn + Ian)),
	Ipa = (double)at(px,ay,z,c), Ica = (double)at(x,ay,z,c), Ina = (double)at(nx,ay,z,c),
	Iaa = (double)at(ax,ay,z,c),
	Ia = Ica + 0.5f*(dx*(-Ipa + Ina) + dx*dx*(2*Ipa - 5*Ica + 4*Ina - Iaa) + dx*dx*dx*(-Ipa + 3*Ica - 3*Ina + Iaa));
	return Ic + 0.5f*(dy*(-Ip + In) + dy*dy*(2*Ip - 5*Ic + 4*In - Ia) + dy*dy*dy*(-Ip + 3*Ic - 3*In + Ia));
}


Image * deviceImageFromCImg(const cimg_library::CImg<unsigned char> & image) {

	Image * hImg = new Image();
	hImg->width = image.width();
	hImg->height = image.height();
	hImg->spectrum = image.spectrum();
	hImg->depth = image.depth();

	const unsigned char * imgData = image.data();
	unsigned char * dImgData;
	cudaMalloc(&dImgData, sizeof(unsigned char) * image.size());
	cudaMemcpy(dImgData, imgData, sizeof(unsigned char) * image.size(), cudaMemcpyHostToDevice);
	hImg->data = dImgData;

	Image * dImg;
	cudaMalloc(&dImg, sizeof(Image));
	cudaMemcpy(dImg, hImg, sizeof(Image), cudaMemcpyHostToDevice);

	delete hImg;

	return dImg;
}