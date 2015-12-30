#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#include "CImg.h"

using namespace cimg_library;
using namespace std;

enum CommunicationTag
{
	COMM_TAG_MASTER_SEND_IMG,
	COMM_TAG_MASTER_SEND_TASK,
	COMM_TAG_MASTER_SEND_TERMINATE,
	COMM_TAG_SLAVE_SEND_RESULT,
};

void sendImage(CImg<char> * img, int proc)
{
	int * headers = new int[4];
	headers[0] = img->width();
	headers[1] = img->height();
	headers[2] = img->depth();
	headers[3] = img->spectrum();
	char * data = img->data();
	int size = img->size();

	MPI_Send(headers, 4, MPI_INT, proc, COMM_TAG_MASTER_SEND_IMG, MPI_COMM_WORLD);
	MPI_Send(data, size, MPI_CHAR, proc, COMM_TAG_MASTER_SEND_IMG, MPI_COMM_WORLD);

	delete[] headers;
}

CImg<char> * receiveImage(int proc)
{
	int headersSize = 0;
	int dataSize = 0;
	MPI_Status stats;
	MPI_Probe(proc, COMM_TAG_MASTER_SEND_IMG, MPI_COMM_WORLD, &stats);
	MPI_Get_count(&stats, MPI_INT, &headersSize);

	if (headersSize != 4)
	{
		return NULL;
	}

	int * imgHeaders = new int[headersSize];

	MPI_Recv(imgHeaders, headersSize, MPI_INT, proc, COMM_TAG_MASTER_SEND_IMG, MPI_COMM_WORLD, &stats);

	int width = imgHeaders[0];
	int height = imgHeaders[1];
	int depth = imgHeaders[2];
	int chans = imgHeaders[3];
	int size = width * height * depth * chans;

	delete[] imgHeaders;

	MPI_Probe(proc, COMM_TAG_MASTER_SEND_IMG, MPI_COMM_WORLD, &stats);
	MPI_Get_count(&stats, MPI_CHAR, &dataSize);

	if (dataSize != size)
	{
		return NULL;
	}

	char * imgData = new char[size];
	MPI_Recv(imgData, size, MPI_CHAR, proc, COMM_TAG_MASTER_SEND_IMG, MPI_COMM_WORLD, &stats);

	CImg<char> *image = new CImg<char>(width, height, depth, chans);
	image->_data = imgData;

	return image;
}

int main(int argc, char *argv[]) 
{
	// 0. Init part, finding rank and number of processes
	//------------------------------------------------------
	int  numprocs, rank, rc;
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) 
	{
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	CImg<char> * image1;
	CImg<char> * image2;

	if (rank == 0) 
	{
		clock_t tStartt = clock();
		image1 = new CImg<char>("important.jpg");
		image2 = new CImg<char>("important2.jpg");
		printf("Time taken to read imgs: %.2fs\n", (double)(clock() - tStartt)/CLOCKS_PER_SEC);

		clock_t tStart = clock();

		printf("Master %d %d %d\n", (*image2)(0,0,0,0), (*image2)(0,0,0,1), (*image2)(0,0,0,2));

		for (int i = 1; i < numprocs;i ++)
		{
			sendImage(image1, i);
			sendImage(image2, i);
		}

		MPI_Status(stats);
		
		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	}
	else
	{
		image1 = receiveImage(0);
		if (image1 == NULL)
		{
			printf("Error when receiving image1. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
		}

		image2 = receiveImage(0);
		if (image2 == NULL)
		{
			printf("Error when receiving image2. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
		}
		printf("Slave %d %d %d %d\n", rank, (*image2)(0,0,0,0), (*image2)(0,0,0,1), (*image2)(0,0,0,2));
	}

	int width = image1->width();
	int height = image1->height();



	MPI_Finalize(); 
}