#ifndef _TIMER_H_
#define _TIMER_H_

#include <sys/time.h>
#include <stdio.h>

typedef struct{
	struct timeval start;
	struct timeval stop;
}Timer;

void startTimer(Timer *pTimer);
void stopTimer(Timer *pTimer);
double getElapsedTime(Timer *pTimer);
void writeTimeToFile(double arrTime[], int nCount, int nFrameNo, char *filename);

#define INITIALIZE_TIMER Timer stTimer; double arrTime[100]
#define START_TIMER startTimer(&stTimer)
#define STOP_TIMER(text) stopTimer(&stTimer); printf(text); printf(": %.1f\n",getElapsedTime(&stTimer))
#define END_TIMER(nIndex) stopTimer(&stTimer); arrTime[nIndex] = getElapsedTime(&stTimer)
#define ACC_TIMER(nIndex) stopTimer(&stTimer); arrTime[nIndex] += getElapsedTime(&stTimer)
#define WRITE_TIME_FILE(nCount, nFrameNo, filename) writeTimeToFile(arrTime, nCount, nFrameNo, (char *)filename)

#endif//_TIMER_H_


