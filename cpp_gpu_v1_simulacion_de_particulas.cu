/*
Versión CUDA C++ para GPU

En esta versión se paralelizan las operaciones de cálculo de fuerzas, integración de las ecuaciones de movimiento, rebote blando y corrección de la temperatura en cada paso de tiempo. Cada cálculo requiere un kernel distinto.
*/

//Importo librerías
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
const float K = 1.380649e-23 * (1 / 1e-7); // constante de Boltzmann [ergio/K], obtenida de NIST


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

__global__
void correccion_Temperatura(Particula *p, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    //Calculo lambda
    float denominador = 0;
    for (int j = 0; j < N; ++j){
        denominador += p[j].vx * p[j].vx + p[j].vy * p[j].vy;
    }
    float lambda = sqrt(N/denominador);
    //Corrijo las temperaturas
    p[i].vx = lambda*p[i].vx;
    p[i].vy = lambda*p[i].vy;

  }
}



int main(const int argc, const char** argv) {

  /***************************************
  CONDICIONES INICIALES
  ****************************************/
  // Radio del círculo y velocidad inicial de la partícula
  const float R0_dim = 1e-6; // [cm]
  const float T0_dim = 1000; // [K]

  //Nro de partículas
  int N = 100000;
  if (argc > 1) N = atoi(argv[1]);

  /***************************************
  CONDICIONES DE INTEGRACIÓN
  ****************************************/
  const float dt = 1e-3; //1e-8;; // time step
  const int n_pasos = 10;
  int guardo_cada = 100;  // Valor deseado para guardo_cada


  /***************************************
  CUENTAS PREVIAS
  ****************************************/

  // Cálculo de las constantes adimensionales
  float v0_dim = sqrt(2 * K * T0_dim / m);
  float alpha = pow(e, 2) / (m * R0_dim * pow(v0_dim, 2));
  cout << "Constante adimensional, alpha = " << alpha << endl;

  // Radio del círculo y velocidad inicial de la partícula adimensionales
  float R0 = 1;
  float v0 = 1;
  float t = 0.;
 

  //Defino archivos para guardar los resultados
  ofstream pos_x_file("resultados/cpp_gpu_v1_pos_x.txt");
  ofstream pos_y_file("resultados/cpp_gpu_v1_pos_y.txt");
  ofstream vel_x_file("resultados/cpp_gpu_v1_vel_x.txt");
  ofstream vel_y_file("resultados/cpp_gpu_v1_vel_y.txt");
  ofstream t_file("resultados/cpp_gpu_v1_t.txt");
  ofstream t_computo_file("resultados/cpp_cpu_t_computo.txt", ios_base::app);

  cout << "Archivos creados correctamente" << endl;

  /***************************************
  ALOCACIÓN DE MEMORIA
  ****************************************/

  //Aloco memoria en host
  int bytes = N*sizeof(Particula);

  float *buf = (float*)malloc(bytes);
  Particula *p = (Particula*)buf;

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

  //Asigno las condiciones iniciales
  condiciones_iniciales(p, N);

  int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double t_computo_total = 0.0; //Tiempo de cómputo


  /***************************************
  LOOP TEMPORAL
  ****************************************/
  for (int iter = 1; iter <= n_pasos; iter++) {
    // En cada loop de tiempo se copian los datos a la GPU, se paraleliza en GPU y luego se vuelven a copiar los datos a CPU  
    StartTimer();

    /***************************************
    MÉTODO DE VERLET
    ****************************************/

    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice); //Esto se puede poner afuera del Loop. TO-DO
    
    //1. Calculo las fuerzas
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt1, dt, N, alpha);
    cudaDeviceSynchronize();    

    //2. Integro las posiciones
    position_integration<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt1, dt, N);
    cudaDeviceSynchronize();  


    //3. Calculo las fuerzas con las nuevas posiciones
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt2, dt, N, alpha);
    cudaDeviceSynchronize();  

    //4. Integro las velocidades
    velocity_integration<<<nBlocks, BLOCK_SIZE>>>(d_p, d_dpdt1, d_dpdt2 , dt, N);
    cudaDeviceSynchronize();  

    //5. Computo los rebotes
    rebote_blando<<<nBlocks, BLOCK_SIZE>>>(d_p, N, R0); 
    cudaDeviceSynchronize();  

    //6. Corrijo las velocidades para que la temperatura sea la deseada
    correccion_Temperatura<<<nBlocks, BLOCK_SIZE>>>(d_p, N);
    cudaDeviceSynchronize();

    /***************************************
    GUARDO DATOS
    ****************************************/
    t = t + dt;

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


    /***************************************
    CALCULO TIEMPOS DE CÓMPUTO
    ****************************************/

    const double t_computo_paso = GetTimer() / 1000.0;
    //No tiro ninguna iteración
    t_computo_total += t_computo_paso;
    if (i == 0){
        t_computo_file << N << " ";
    }
    t_computo_file << t_computo_paso << " ";
    if (i == n_pasos - 1){
        t_computo_file << "\n";
    }


  

    #ifndef SHMOO
        printf("Iteration %d: %.3f seconds\n", iter, t_computo_paso);
    #endif
  }

  //Cierro archivos
  pos_x_file.close();
  pos_y_file.close();
  vel_x_file.close();
  vel_y_file.close();
  t_file.close();
  t_computo_file.close();
  

  //Guardo condiciones iniciales
  ofstream cond_ini_file("resultados/cpp_gpu_v1_cond_ini.txt");
  cond_ini_file << R0 << " " << v0 << " " << R0_dim << " " << v0_dim;


  double avgTime = t_computo_total / (double)(n_pasos-1); 

  #ifdef SHMOO
    printf("%d, %0.3f\n", N, 1e-9 * N * N / avgTime);
  #else
    //printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per second.\n",
    //       n_pasos, rate);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", N, 1e-9 * N * N / avgTime);
  #endif
    free(buf);
    cudaFree(d_buf);
    cudaFree(d_buf_dt1);
    cudaFree(d_buf_dt2);
}
