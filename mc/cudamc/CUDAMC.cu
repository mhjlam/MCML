/////////////////////////////////////////////////////////////
//
//		CUDA-based Monte Carlo simulation of photon migration in semi infinite media.
//	
//			This is the version of the code used in the letter submitted to JBO-letters 2008
//			Currently the code is in an experimental state, i.e. the code is not always pretty 
//			or efficient and some ways of implementing certain aspects of the code are far 
//			from desirable. Still it should provide a good starting point for anyone interested 
//			in CUDA-based Monte Carlo simulations of photon migration. 
//
//			For the JBO-letters article the code was run on a NVIDIA 8800GT and the number of 
//			threads are hence optimized for this particular card.
//
//			We apologize for the lack of comment in the current code. We will soon re-relese 
//			this code with detailed explanations of the implementation as well as proper commenting.
//
//			To compile and run this code, please visit www.nvidia.com and download the necessary 
//			CUDA Toolkit and SKD. I also highly recommend the Visual Studio wizard 
//			(available at:http://forums.nvidia.com/index.php?showtopic=69183) 
//			if you use Visual Studio 2005 
//			(The express edition is available for free at: http://www.microsoft.com/express/2005/). 
//
//			This code is distributed under the terms of the GNU General Public Licence (see
//			below). If you use this code for academic purposes, we would greatly appreciate a 
//			citation of our letter describing GPU-based Monte Carlo simulations of photon migration. 
//
//
///////////////////////////////////////////////////////////////

/*	This file is part of CUDAMC.

    CUDAMC is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMC is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMC.  If not, see <http://www.gnu.org/licenses/>.*/


#define NUM_THREADS_PER_BLOCK 256 // Conservative value that works on all GPUs
#define NUM_BLOCKS 64 // Conservative number of blocks
#define NUM_THREADS 16384 // NUM_BLOCKS * NUM_THREADS_PER_BLOCK = 64 * 256
#define NUMSTEPS_GPU 500000
#define NUMSTEPS_CPU 500000
#define PI 3.14159265f

#define TMAX 2000.0f //[ps] Maximum time of flight
#define DT 10.0f //[ps] Time binning resolution
#define TEMP 201 //ceil(TMAX/DT), precalculated to avoid dynamic memory allocation (fulhack)


unsigned int xtest[NUM_THREADS];
unsigned int ctest[NUM_THREADS];
unsigned int atest[NUM_THREADS];

// forward declaration of the device code
__global__ void MCd(unsigned int*,unsigned int*,unsigned int*,unsigned int*,unsigned int*);
__device__ float rand_MWC_oc(unsigned long long*,unsigned int*);
__device__ float rand_MWC_co(unsigned long long*,unsigned int*);
__device__ void LaunchPhoton(float3*, float3*, float*);
__device__ void Spin(float3*,float*,unsigned long long*,unsigned int*);
__device__ unsigned int Reflect(float3*, float3*, float*, float*, float*, float*, unsigned long long*,unsigned int*,unsigned int*);

// forward declaration of the host code
void MCh(unsigned int*,unsigned int*,unsigned int*,unsigned int*,unsigned int*);
float rand_MWC_och(unsigned long long*,unsigned int*);
float rand_MWC_coh(unsigned long long*,unsigned int*);
void LaunchPhotonh(float3*, float3*, float*);
void Spinh(float3*,float*,unsigned long long*,unsigned int*);
unsigned int Reflecth(float3*, float3*, float*, float*, float*, float*, unsigned long long*,unsigned int*,unsigned int*);

#include "math_constants.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CUDAMCtransport.cu"
#include "CUDAMC_goldstandard.c"

// wrapper for device code
void MC(unsigned int* x,unsigned int* c,unsigned int* a)
{
	// Check CUDA device availability
	int deviceCount;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess || deviceCount == 0) {
		printf("No CUDA devices found or CUDA error: %d\n", error);
		return;
	}
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Using CUDA device: %s\n", prop.name);
	printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max blocks per grid: %d\n", prop.maxGridSize[0]);
	
	unsigned int num[NUM_THREADS];
	unsigned long long num_tot, hist_tot;
	unsigned int i;
	unsigned int hist[TEMP];

	unsigned int numh[NUM_THREADS];
	unsigned int histh[TEMP];

	
    cudaError_t cudastat;
    clock_t time1,time2,GPUtime,CPUtime;
    int size;
    size=NUM_THREADS*sizeof(unsigned int);

    time1=clock();


    //load initial values
    unsigned int* xd;
    cudaMalloc((void**)&xd,size);
    cudaMemcpy(xd,xtest,size,cudaMemcpyHostToDevice);
	
    unsigned int* cd;
    cudaMalloc((void**)&cd,size);
    cudaMemcpy(cd,ctest,size,cudaMemcpyHostToDevice);
	
    unsigned int* ad;
    cudaMalloc((void**)&ad,size);
    cudaMemcpy(ad,atest,size,cudaMemcpyHostToDevice);
	
    //allocate numd on the device
	unsigned int* numd;
    cudaMalloc((void**)&numd,size);

	for(i=0;i<TEMP;i++)hist[i]=0;
	unsigned int* histd;
    cudaMalloc((void**)&histd,size);
	cudaMemcpy(histd,hist,TEMP*sizeof(unsigned int),cudaMemcpyHostToDevice);

    dim3 dimBlock(NUM_THREADS_PER_BLOCK);
    dim3 dimGrid(NUM_BLOCKS);
	cudaDeviceSynchronize(); //probably not necessary


    MCd<<<dimGrid,dimBlock>>>(xd,cd,ad,numd,histd);
    
    // Check for kernel launch errors immediately
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(kernelError));
        return;
    }
    
    // Check for kernel execution errors
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(syncError));
        return;
    }



	cudaMemcpy(num,numd,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(hist,histd,TEMP*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize(); //probably not necessary

    cudastat=cudaGetLastError();
	printf("\nError code=%i\n",cudastat);
    printf("Error code=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));

	//Free device memory
    cudaFree(xd);
    cudaFree(cd);
    cudaFree(ad);
	cudaFree(numd);
	cudaFree(histd);

    time2=clock();

	num_tot=0;
	for(i=0;i<NUM_THREADS;i++)num_tot+=num[i];

	hist_tot=0;
	for(i=0;i<TEMP;i++)hist_tot+=hist[i];
	for(i=0;i<TEMP;i++)printf("%d ",hist[i]);

	FILE *file;
	file = fopen("outp.txt", "w");
	for(i=0;i<TEMP;i++)fprintf(file,"%d ",hist[i]);
	fclose(file);
	printf("\nTotal number of photons terminated (i.e. full path simulated): %llu\nNumber of photons contribution to the histogram: %llu\n",num_tot,hist_tot);
	printf("Total number of photons steps simulated: %e\n",(double)NUM_THREADS*(double)NUMSTEPS_GPU);
    printf("time1=%u, time2=%u.\n",time1,time2);

	printf("Photon steps per sec: %e\n",((double)NUM_THREADS*(double)NUMSTEPS_GPU)*1000/(double(time2-time1)));
	GPUtime=time2-time1;


	printf("\n\nRunning CPU code\n");

	for(i=0;i<TEMP;i++)histh[i]=0;

	//run CPU code
	time1=clock();
	MCh(xtest,ctest,atest,numh,histh);
    time2=clock();

	num_tot=0;
	for(i=0;i<NUM_THREADS;i++)num_tot+=numh[i];

	hist_tot=0;
	for(i=0;i<TEMP;i++)hist_tot+=histh[i];
	for(i=0;i<TEMP;i++)printf("%d ",histh[i]);

	file = fopen("outph.txt", "w");
	for(i=0;i<TEMP;i++)fprintf(file,"%d ",histh[i]);
	fclose(file);
	printf("\n\nTotal number of photons (i.e. full path simulated): %llu\nNumber of photons contribution to the histogram: %llu\n",num_tot,hist_tot);
	printf("Total number of photons steps simulated: %e\n",(double)NUM_THREADS*(double)NUMSTEPS_CPU);
    printf("time1=%u, time2=%u.\n",time1,time2);

	printf("Photon steps per sec: %e\n",((double)NUM_THREADS*(double)NUMSTEPS_CPU)*1000/(double(time2-time1)));
	CPUtime=time2-time1;

	printf("\n\nSpeedup: %f",(NUMSTEPS_GPU*double(CPUtime))/(NUMSTEPS_CPU*double(GPUtime)));
}




void initialize(void)//Straight from Steven Gratton's code
{
    FILE *fp;
    unsigned int begin=0u;
    unsigned long long int xinit=1ull;
    unsigned int cinit=0u;
    unsigned int fora,tmp1,tmp2;
    fp=fopen("safeprimes_base32.txt","r");//use an expanded list containing 50000 safeprimes instead of Steven's shorter list


// use begin as a multiplier to generate the initial x's for 
// the other generators...

    fscanf(fp,"%u %u %u",&begin,&tmp1,&tmp2);

    for (unsigned int i=0;i<NUM_THREADS;i++)
    {

	xinit=xinit*begin+cinit;
	cinit=xinit>>32;
	xinit=xinit&0xffffffffull;
	xtest[i]=(unsigned int) xinit;
	fscanf(fp,"%u %u %u",&fora,&tmp1,&tmp2);
	atest[i]=fora;

	xinit=xinit*begin+cinit;
	cinit=xinit>>32;
	xinit=xinit&0xffffffffull;
	ctest[i]=(unsigned int) ((((double)xinit)/UINT_MAX)*fora);

    }
    fclose(fp);
}



int main(int argc,char* argv[])
{
	//do all the initialization for the RNG's (one MWC per thread)
    initialize();
    MC(xtest,ctest,atest);
	return 0;
    
}
