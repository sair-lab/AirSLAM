#include <stdlib.h>

#include "timer.h"


void startTimer(Timer *pTimer){
	gettimeofday(&pTimer->start, NULL);
}
void stopTimer(Timer *pTimer){
	gettimeofday(&pTimer->stop, NULL);
}

double getElapsedTime(Timer *pTimer){
	return ((pTimer->stop.tv_sec - pTimer->start.tv_sec) * 1000.0 + (pTimer->stop.tv_usec - pTimer->start.tv_usec) / 1000.0);
}

void writeTimeToFile(double arrTime[], int nCount, int nFrameNo, char *filename){
	FILE *pFile;
	int i;

	pFile = fopen(filename, "a+");
	if(NULL == pFile)
		return;

	fprintf(pFile, "FrameNo, %d", nFrameNo);

	for (i = 0; i < nCount; i++){
		fprintf(pFile, ",%f", arrTime[i]);
	}

	fprintf(pFile, "\n");
	fclose(pFile);
}
