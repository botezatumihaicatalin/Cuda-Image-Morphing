#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "CImg.h"
#include <time.h>

using namespace cimg_library;
using namespace std;

enum CommunicationTag
{
	COMM_TAG_MASTER_SEND_TASK,
	COMM_TAG_MASTER_SEND_TERMINATE,
	COMM_TAG_SLAVE_SEND_RESULT,
};

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

	if (rank == 0) 
	{
		clock_t tStartt = clock();
		CImg<char> image1("important.jpg");
		CImg<char> image2("important2.jpg");

		printf("Image1 : %d %d\n", image1.height(), image1.width());
		
		CImg<char> output(image1);
		output.fill(1);
		printf("Time taken to load imgs: %.2fs\n", (double)(clock() - tStartt)/CLOCKS_PER_SEC);

		clock_t tStart = clock();
		
		int height = output.height();
		int width = output.width();

		int totalPixels = width * height;
		int maxPixelsPerProc = numprocs == 1 ? totalPixels : (totalPixels / (numprocs - 1)) + 1;
		
		int x, y;
		char * rgb = new char[3];
		int currPixelIndex = 0;

		for (int procRank = 1; procRank < numprocs; procRank ++)
		{
			int sendPixels = totalPixels > maxPixelsPerProc ? maxPixelsPerProc : totalPixels;
			totalPixels -= sendPixels;

			MPI_Send(&sendPixels, 1, MPI_INT, procRank, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			printf("Sending %d\n", sendPixels);

			vector<int> xyBuf;
			vector<char> rgbBuf;

			int addedPixels = 0;
			for (;currPixelIndex < width * height && addedPixels < sendPixels; currPixelIndex++)
			{
				x = currPixelIndex % width;
				y = currPixelIndex / width;
				xyBuf.push_back(x);
				xyBuf.push_back(y);

				for (int c = 0; c < 3; c++)
				{
					rgbBuf.push_back(image1(x, y, 0, c));
				}
				addedPixels ++;
			}

			MPI_Send(&xyBuf.front(), xyBuf.size(), MPI_INT, procRank, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			MPI_Send(&rgbBuf.front(), rgbBuf.size(), MPI_CHAR, procRank, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
		}

		MPI_Status(status);

		for (int i = 1; i < numprocs; i++)
		{
			int slaveCount = 0;
			MPI_Recv(&slaveCount, 1, MPI_INT, i, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD, &status);
			
			printf("Received on master count: %d\n", slaveCount);

			vector<int> xyBuf;
			vector<char> rgbBuf;

			xyBuf.resize(slaveCount * 2);
			rgbBuf.resize(slaveCount * 3);

			MPI_Recv(&xyBuf.front(), xyBuf.size(), MPI_INT, i, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD, &status);
			MPI_Recv(&rgbBuf.front(), rgbBuf.size(), MPI_CHAR, i, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD, &status);

			for (int index = 0; index < slaveCount; index++)
			{
				for (int c = 0; c < 3; c++)
				{
					output(xyBuf[2 * index], xyBuf[2 * index + 1], 0, c) = rgbBuf[3 * index + c];
				}
				//printf("Received on master %d %d %d %d %d\n",xyBuf[index * 2], xyBuf[index * 2 + 1], rgbBuf[3 * index], rgbBuf[3 * index + 1], rgb[3 * index + 2]);
			}
		}

		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

		output.save("output.jpg");
	}
	else
	{
	    MPI_Status status;
		MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		
		int count;
		vector<int> xyBuf;
		vector<char> rgbBuf;
		
		MPI_Recv(&count, 1, MPI_INT, 0, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &status);

		xyBuf.resize(count * 2);
		rgbBuf.resize(count * 3);

		MPI_Recv(&xyBuf.front(), xyBuf.size(), MPI_INT, 0, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &status);
		MPI_Recv(&rgbBuf.front(), rgbBuf.size(), MPI_CHAR, 0, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &status);

		for (int i = 0; i < count; i ++)
		{
			rgbBuf[3 * i] = -rgbBuf[3 * i];
			rgbBuf[3 * i + 1] = -rgbBuf[3 * i + 1];
			rgbBuf[3 * i + 2] = -rgbBuf[3 * i + 2];
		}

		MPI_Send(&count, 1, MPI_INT, 0, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD);
		MPI_Send(&xyBuf.front(), xyBuf.size(), MPI_INT, 0, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD);
		MPI_Send(&rgbBuf.front(), rgbBuf.size(), MPI_CHAR, 0, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}