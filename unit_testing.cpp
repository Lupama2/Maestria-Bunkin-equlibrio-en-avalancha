// Importo librerías

#include <iostream>
#include <cmath>

#include "funciones.h"

using namespace std;

int main() {
    float error = false;
    int N = 10; //nro de partículas a simular en cada test

    //Test sobre f_maxwell
    for(int i = 0; i < N; i++){
        double f = f_maxwell();
        if (f < 0 || f > 1){
            cout << "Error en f_maxwell" << endl;
            error = true;
            break;
        }
    }
    if (error == false){
        cout << "f_maxwell en (0,1)" << endl;}

    //Test sobre condiciones iniciales
    error = false;
    double* y0 = new double[4 * N];
    condiciones_iniciales(y0);
    for(int n = 0; n < N; n++){
        //Controlo que estén inicialmente dentro del círculo de radio R0
        double R0 = 1.0;
        double r = sqrt(pow(y0[n],2) + pow(y0[N + n],2));
        if (r > R0){
            cout << "Error en posiciones iniciales" << endl;
            error = true;
            break;
        }


        //Controlo que la velocidad esté entre 0 y v0
        double v0 = 1.0;
        double v = sqrt(pow(y0[2*N + n],2) + pow(y0[3*N + n],2));
        if (v < 0 || v > v0){
            cout << "Error en velocidades iniciales" << endl;
            cout << v << endl;
            error = true;
            break;
        }
    }
    if (error == false){
        cout << "Condiciones iniciales dentro del circulo de radio R0 y con velocidad entre 0 y v0" << endl;}

    //Test sobre distancia_al_origen
    // void distancia_al_origen(double* r_vec, double* d_vec);
    error = false;
    double* r_vec = new double[2 * N];
    double* d_vec = new double[N];
    for(int n = 0; n < N; n++){
        r_vec[n] = 1.0;
        r_vec[N + n] = 0.0;
    }
    distancia_al_origen(r_vec, d_vec);
    for(int n = 0; n < N; n++){
        if (d_vec[n] != 1.0){
            cout << "Error en distancia_al_origen" << endl;
            error = true;
            break;
        }
    }
    if (error == false){
        cout << "distancia_al_origen funciona correctamente" << endl;}
    
    //Test sobre rebote_blando
    // void rebote_blando(double rx, double ry, double vx, double vy, double *result);
    error = false;
    double* result = new double[4];
    double epsilon = 1e-15; //error permitido en la comparación de doubles
    rebote_blando(1.1, 0.0, 1.0, 0.0, result);
    if (result[0] != 1.1 || result[1] != 0.0 || result[2] != -1.0 || result[3] != 0.0){
        cout << "Error en rebote_blando, caso 1" << endl;
        error = true;
    }

    rebote_blando(0.0, -1.1, 0.0, -1.0, result);
    if (result[0] != 0.0 || result[1] != -1.1 || (result[2] - 0.0) > epsilon || result[3] != 1.0){
        cout << "Error en rebote_blando, caso 2" << endl;
        // cout << result[0] << "\t" << result[1] << "\t"<< result[2] << "\t"<< result[3] << endl;
        error = true;
    }

    if (error == false){
        cout << "rebote_blando funciona correctamente" << endl;}
    



    return 0;
}

