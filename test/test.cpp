/*
Para compilar
g++ test.cpp ../cpp_cpu_funciones.cpp -o test



*/


// Importo librerías
#include <iostream>
#include <cmath>
#include <random>
//Para nros random:
#include <cstdlib>
// #include <ctime>
#include "../cpp_cpu_funciones.h"
using namespace std;


// Accedo a la función f_maxwell de cpp_cpu_funciones.cpp





int main(){

    //Test 1: genero muchos nros con f_maxwell_pura y los guardo en un arhcivo 
    bool test1 = false;
    if(test1 == true){
        int N = 1000000;
        float* v_vec = new float[N];
        
        for(int i = 0; i < N; ++i){
            v_vec[i] = f_maxwell_pura();
        }

        // Guardo en un archivo
        FILE *fp;
        fp = fopen("test1.txt", "w");
        for(int i = 0; i < N; ++i){
            fprintf(fp, "%f\n", v_vec[i]);
        }
        fclose(fp);
    }

    //Test 2
    bool test2 = true;
    if(test2 == true){
        int N = 1000000;
        float* v_vec = new float[N];
        
        for(int i = 0; i < N; ++i){
            // float T = 100; //[K]
            // float m = 9.11e-31*1e3; //[g]
            // float K = 1.380649e-23*(1/1e-7); //constante de Boltzmann [ergio/K], obtenida de NIST
            // float v0_dim = sqrt(2 * K * T / m);
            v_vec[i] = f_maxwell_adim();
        }

        // Guardo en un archivo
        FILE *fp;
        fp = fopen("test2.txt", "w");
        for(int i = 0; i < N; ++i){
            fprintf(fp, "%f\n", v_vec[i]);
        }
        fclose(fp);
    }

    return 0;
}