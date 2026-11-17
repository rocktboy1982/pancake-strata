#include <mma.h>
#include <cstdio>
#include <cuda_fp16.h>
using namespace nvcuda;
#define ITERS 4096
__global__ void k_f16(float* o){ wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a; wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b; wmma::fragment<wmma::accumulator,16,16,16,float> c;
 wmma::fill_fragment(a,__float2half(0.01f)); wmma::fill_fragment(b,__float2half(0.02f)); wmma::fill_fragment(c,0.f);
 for(int i=0;i<ITERS;i++) wmma::mma_sync(c,a,b,c); if(c.x[0]==12345.f) o[0]=c.x[0]; }
__global__ void k_s8(int* o){ wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> a; wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major> b; wmma::fragment<wmma::accumulator,16,16,16,int> c;
 wmma::fill_fragment(a,1); wmma::fill_fragment(b,2); wmma::fill_fragment(c,0);
 for(int i=0;i<ITERS;i++) wmma::mma_sync(c,a,b,c); if(c.x[0]==12345) o[0]=c.x[0]; }
__global__ void k_dp4a(int* o){ int a=threadIdx.x,b=3,acc[8]={0}; for(int i=0;i<ITERS*8;i++){ for(int j=0;j<8;j++) acc[j]=__dp4a(a+j,b+i,acc[j]); } int s=0; for(int j=0;j<8;j++)s+=acc[j]; if(s==12345) o[0]=s; }
__global__ void k_f32(float* o){ float a=threadIdx.x*1e-3f,acc[8]={0}; for(int i=0;i<ITERS*8;i++){ for(int j=0;j<8;j++) acc[j]=fmaf(a+j,1.0001f,acc[j]); } float s=0; for(int j=0;j<8;j++)s+=acc[j]; if(s==12345.f) o[0]=s; }
__global__ void k_h2(float* o){ half2 a=__float2half2_rn(threadIdx.x*1e-3f),acc[8]; for(int j=0;j<8;j++)acc[j]=__float2half2_rn(0.f); half2 m=__float2half2_rn(1.0001f); for(int i=0;i<ITERS*8;i++){ for(int j=0;j<8;j++) acc[j]=__hfma2(a,m,acc[j]); } float s=0; for(int j=0;j<8;j++)s+=__low2float(acc[j]); if(s==12345.f) o[0]=s; }
template<class F> float tmr(F f){ cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e); f(); cudaDeviceSynchronize(); cudaEventRecord(s); f(); cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e); return ms; }
int main(){ int sm; cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0); int blocks=sm*8, thr=128, warps=blocks*thr/32; void* o; cudaMalloc(&o,64);
 float ms=tmr([&]{k_f16<<<blocks,thr>>>((float*)o);}); printf("HMMA f16 16x16x16: %.2f TFLOPS (sm=%d)\n", 2.0*16*16*16*ITERS*warps/ms/1e9, sm);
 ms=tmr([&]{k_s8<<<blocks,thr>>>((int*)o);}); printf("IMMA s8  16x16x16: %.2f TOPS\n", 2.0*16*16*16*ITERS*warps/ms/1e9);
 ms=tmr([&]{k_dp4a<<<blocks,thr>>>((int*)o);}); printf("dp4a: %.2f TOPS\n", 2.0*4*ITERS*8.0*blocks*thr/ms/1e9);
 ms=tmr([&]{k_f32<<<blocks,thr>>>((float*)o);}); printf("fp32 fma: %.2f TFLOPS\n", 2.0*ITERS*8.0*blocks*thr/ms/1e9);
 ms=tmr([&]{k_h2<<<blocks,thr>>>((float*)o);}); printf("half2 fma: %.2f TFLOPS\n", 4.0*ITERS*8.0*blocks*thr/ms/1e9);
 printf("err %s\n", cudaGetErrorString(cudaGetLastError())); }
