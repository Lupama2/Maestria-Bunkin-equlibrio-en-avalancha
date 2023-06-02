#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"


using namespace std;

#define BLOCK_SIZE 256
#define SOFTENING 1e-14f

typedef struct { float x, y, vx, vy; } Particula;

//Constantes matemáticas
const float pi = acos(-1.0);

// Constantes físicas
const float m = 9.11e-31 * 1e3; // [g]
const float e = 1.602e-19 * (1 / 3.336641e-10); // [Fr]
// const float c = 299792458 * 1e2; // [cm/s]
const float K = 1.380649e-23 * (1 / 1e-7); // constante de Boltzmann [ergio/K], obtenida de NIST

// Radio del círculo y velocidad inicial de la partícula
const float R0_dim = 1e-6; // [cm]
const float T0_dim = 300; // [K]


void condiciones_iniciales(Particula *p0, int N) {
  //Lo siguiente es válido por cómo fue definida la adimnesionalización
  float R0 = 1.0;

  for (int i = 0; i < N; ++i){
    float r0 = R0 * (rand() / (float)RAND_MAX);
    float tita_r0 = (rand() / (float)RAND_MAX);
    p0[i].x = r0 * cos(tita_r0); // = rx0_vec[i]
    p0[i].y = r0 * sin(tita_r0); // = ry0_vec[i] 

    float v0 = (rand() / (float)RAND_MAX); //Por ahora va a ser random uniforme. TO-DO
    float tita_v0 = (rand() / (float)RAND_MAX);
    p0[i].vx = v0 * cos(tita_v0); // = vx0_vec[i]
    p0[i].vy = v0 * sin(tita_v0); // = vy0_vec[i]
  }
}

__global__
void bodyForce(Particula *p, Particula *dpdt, float dt, int N, float alpha) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    float Fx = 0.0f; float Fy = 0.0f;

    for (int j = 0; j < N; j++) {
      float dx = p[i].x - p[j].x;
      float dy = p[i].y - p[j].y;
      float r2 = dx*dx + dy*dy + SOFTENING;
      float inv_r = rsqrtf(r2);
      float inv_r3 = inv_r * inv_r * inv_r;

      Fx += alpha * dx * inv_r3; Fy += alpha * dy * inv_r3;
    }
    //Asigno las derivadas
    dpdt[i].x = p[i].vx; dpdt[i].y = p[i].vy;
    dpdt[i].vx = Fx; dpdt[i].vy = Fy;
     
  }
}


__global__
void position_integration(Particula *p, Particula *dpdt1, float dt, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    p[i].x = p[i].x + p[i].vx * dt + 0.5 * dt * dt * dpdt1[i].vx;
    p[i].y = p[i].y + p[i].vy * dt + 0.5 * dt * dt * dpdt1[i].vy;
     
  }
}


__global__
void velocity_integration(Particula *p, Particula *dpdt1, Particula *dpdt2 ,float dt, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    // ynew[2*N + i] = yold[2 * N + i] + 0.5 * dt * (F_vec[i] + F_vec_new[i]);
      p[i].vx = p[i].vx + 0.5 * dt *( dpdt1[i].vx + dpdt2[i].vx);
      p[i].vy = p[i].vy + 0.5 * dt *( dpdt1[i].vy + dpdt2[i].vy);
  }
}

__global__
void rebote_blando(Particula *p, int N, float R0) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
      //Opero sobre las partículas que chocaron
      if (sqrt(p[i].x * p[i].x + p[i].y * p[i].y) > R0) {

        // Obtengo las variables correspondientes
        float vx = p[i].vx;
        float vy = p[i].vy;

        float tita = atan2(p[i].y, p[i].x);
        p[i].vx = -vx * cos(2 * tita) - vy * sin(2 * tita);
        p[i].vy = -vx * sin(2 * tita) + vy * cos(2 * tita);
      }
  }
}


int main(const int argc, const char** argv) {
  // int N = 30000; // Nro de partículas del código de ejemplo
  int N = 10;
  if (argc > 1) N = atoi(argv[1]);

  // Cálculo de las constantes adimensionales
  float v0_dim = sqrt(3 * K * T0_dim / m);
  float alpha = pow(e, 2) / (m * R0_dim * pow(v0_dim, 2));
  cout << "Constante adimensional, alpha = " << alpha << endl;

  // Radio del círculo y velocidad inicial de la partícula adimensionales
  float R0 = 1;
  float v0 = 1;

  cout << "Radio y velocidad iniciales adimensionales: " << R0 << ",\t" << v0 << endl;
  

  const float dt = 1e-3; //1e-8;; // time step
  // const int nIters = 10;  // simulation iterations en el ejemplo
  const int n_pasos = 2*3000;
  float t = 0.;

  //Defino archivos para guardar los resultados
  int guardo_cada = 1;  // Valor deseado para guardo_cada

  ofstream pos_x_file("resultados/cpp_gpu_pos_x.txt");
  ofstream pos_y_file("resultados/cpp_gpu_pos_y.txt");
  ofstream vel_x_file("resultados/cpp_gpu_vel_x.txt");
  ofstream vel_y_file("resultados/cpp_gpu_vel_y.txt");
  ofstream t_file("resultados/cpp_gpu_t.txt");

  cout << "Archivos creados correctamente" << endl;

  //Aloco memoria en host
  int bytes = N*sizeof(Particula);

  float *buf = (float*)malloc(bytes);
  Particula *p = (Particula*)buf;
  
  // float *buf_dt1 = (float*)malloc(bytes);
  // Particula *dpdt1 = (Particula*)buf_dt1;

  // float *buf_dt2 = (float*)malloc(bytes);
  // Particula *dpdt2 = (Particula*)buf_dt2;

  //Aloco memoria en device
  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  Particula *d_p = (Particula*)d_buf;

  float *d_buf_dt1;
  cudaMalloc(&d_buf_dt1, bytes);
  Particula *d_dpdt1 = (Particula*)d_buf_dt1;

  float *d_buf_dt2;
  cudaMalloc(&d_buf_dt2, bytes);
  Particula *d_dpdt2 = (Particula*)d_buf_dt2;


  condiciones_iniciales(p, N); // Init pos / vel data


  int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

  for (int iter = 1; iter <= n_pasos; iter++) {
    // En cada loop de tiempo se copian los datos a la GPU, se paraleliza en GPU y luego se vuelven a copiar los datos a CPU  
    StartTimer();


    /**********************************************************/
    //Método de Verlet
    /**********************************************************/

    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    

    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt1, dt, N, alpha); // compute interbody forces 
    cudaDeviceSynchronize();    


    position_integration<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt1, dt, N);
    cudaDeviceSynchronize();  


    // Cálculo de la fuerza en el siguiente paso de tiempo


    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt2, dt, N, alpha); // compute interbody forces 
    cudaDeviceSynchronize();  



    velocity_integration<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt1, d_dpdt2 , dt, N);
    cudaDeviceSynchronize();  


    rebote_blando<<<nBlocks, BLOCK_SIZE>>>(d_p, N, R0); 
    cudaDeviceSynchronize();  

    /**********************************************************/
    //Guardo datos
    /**********************************************************/
    t = t + dt;


    // if (iter % guardo_cada == 0) {
    //   cout << "t = " << t << "\tEvolucion al " << float(i) / float(n_pasos) * 100. << "%\n";
    // }
    if (iter % guardo_cada == 0) {
      //Copio de Device a Host
      cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; i++) {
        pos_x_file << p[i].x << " ";
        pos_y_file << p[i].y << " ";
        vel_x_file << p[i].vx << " ";
        vel_y_file << p[i].vy << " ";
      }
    pos_x_file << "\n";
    pos_y_file << "\n";
    vel_x_file << "\n";
    vel_y_file << "\n";
    t_file << t << "\n";
    }


    /**********************************************************/
    //Calculo tiempos
    /**********************************************************/

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
        totalTime += tElapsed; 
    }

  

    #ifndef SHMOO
        printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
    #endif
  }

  //Cierro archivos
  pos_x_file.close();
  pos_y_file.close();
  vel_x_file.close();
  vel_y_file.close();

  //Guardo condiciones iniciales
  ofstream cond_ini_file("resultados/cpp_gpu_cond_ini.txt");
  cond_ini_file << R0 << " " << v0 << " " << R0_dim << " " << v0_dim;


  double avgTime = totalTime / (double)(n_pasos-1); 

  #ifdef SHMOO
    printf("%d, %0.3f\n", N, 1e-9 * N * N / avgTime);
  #else
    //printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per second.\n",
    //       n_pasos, rate);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", N, 1e-9 * N * N / avgTime);
  #endif
    free(buf);
    cudaFree(d_buf);

}
