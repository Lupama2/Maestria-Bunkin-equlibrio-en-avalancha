#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct { float x, y, vx, vy; } Particula;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(Particula *p, Particula *dpdt, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float r2 = dx*dx + dy*dy + SOFTENING;
      float inv_r = rsqrtf(r2);
      float inv_r3 = inv_r * inv_r * inv_r;

      Fx += dx * inv_r3; Fy += dy * inv_r3;
    }
    //Asigno las derivadas
    dpdt[i].x = p[i].vx; dpdt[i].y = p[i].vy;
    dpdt[i].vx = Fx; dpdt[i].vy = Fy;
     
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations


  //Aloco memoria en host
  int bytes = nBodies*sizeof(Particula);

  float *buf = (float*)malloc(bytes);
  Particula *p = (Particula*)buf;
  
  float *buf_dt = (float*)malloc(bytes);
  Particula *dpdt = (Particula*)buf_dt;

  //Aloco memoria en device
  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  Particula *d_p = (Particula*)d_buf;

  float *d_buf_dt;
  cudaMalloc(&d_buf_dt, bytes);
  Particula *d_dpdt = (Particula*)d_buf_dt;

  randomizeBodies(buf, 4*nBodies); // Init pos / vel data


  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

  for (int iter = 1; iter <= nIters; iter++) {
    // En cada loop de tiempo se copian los datos a la GPU, se paraleliza en GPU y luego se vuelven a copiar los datos a CPU  
    StartTimer();

    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_buf_dt, buf_dt, bytes, cudaMemcpyHostToDevice);

    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt, dt, nBodies); // compute interbody forces 
    
  
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(buf_dt, d_buf_dt, bytes, cudaMemcpyDeviceToHost);


    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].vx += dt*dpdt[i].vx; p[i].vy += dt*dpdt[i].vy;
      p[i].x += dpdt[i].x*dt;
      p[i].y += dpdt[i].y*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
#ifndef SHMOO
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
#endif
  }
  double avgTime = totalTime / (double)(nIters-1); 

#ifdef SHMOO
  printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#else
  //printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per second.\n",
  //       nIters, rate);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#endif
  free(buf);
  cudaFree(d_buf);
}
