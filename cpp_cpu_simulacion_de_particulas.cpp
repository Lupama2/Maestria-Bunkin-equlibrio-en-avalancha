
#include <iostream>
#include <fstream>
#include <cmath>
#include "funciones.h"

//Constantes matemáticas
const double pi = acos(-1.0);

// Constantes físicas
const double m = 9.11e-31 * 1e3; // [g]
const double e = 1.602e-19 * (1 / 3.336641e-10); // [Fr]
const double c = 299792458 * 1e2; // [cm/s]
const double K = 1.380649e-23 * (1 / 1e-7); // constante de Boltzmann [ergio/K], obtenida de NIST

// Radio del círculo y velocidad inicial de la partícula
const double R0_dim = 1e-6; // [cm]
const double T0_dim = 300; // [K]

// Nro de partículas
const int N = 3; 

using namespace std;

int main() {
    // Cálculo de las constantes adimensionales
    double v0_dim = sqrt(3 * K * T0_dim / m);
    double alpha = pow(e, 2) / (m * R0_dim * pow(v0_dim, 2));
    cout << "Constante adimensional, alpha = " << alpha << endl;

    // Radio del círculo y velocidad inicial de la partícula adimensionales
    double R0 = 1;
    double v0 = 1;

    cout << "Radio y velocidad iniciales adimensionales: " << R0 << ",\t" << v0 << endl;

    //Declaro array de datos
    double y[4 * N];
    double ynew[4 * N];

    // Condiciones iniciales
    condiciones_iniciales(y, N);

    double t = 0;
    double dt =  1e-2; //1e-8;
    int n_pasos = 300;

    int guardo_cada = 1;  // Valor deseado para guardo_cada

    ofstream pos_x_file("resultados/cpp_pos_x.txt");
    ofstream pos_y_file("resultados/cpp_pos_y.txt");
    ofstream vel_x_file("resultados/cpp_vel_x.txt");
    ofstream vel_y_file("resultados/cpp_vel_y.txt");
    ofstream t_file("resultados/cpp_t.txt");
    
    cout << "Archivos creados correctamente" << endl;

    for (int i = 0; i < n_pasos; i++) {
        
        if (i % guardo_cada == 0) {
            cout << "t = " << t << "\tEvolucion al " << float(i) / float(n_pasos) * 100. << "%\n";
        }
        
        t += dt;

        if (i % guardo_cada == 0) {
            for (int j = 0; j < N; j++) {
                pos_x_file << y[j] << " ";
                pos_y_file << y[N + j] << " ";
                vel_x_file << y[2 * N + j] << " ";
                vel_y_file << y[3 * N + j] << " ";
            }
            pos_x_file << "\n";
            pos_y_file << "\n";
            vel_x_file << "\n";
            vel_y_file << "\n";
            t_file << t << "\n";
        }
        
        avanzo_dt(y, ynew, t, dt, N, alpha);
        //Copio los datos a y
        for (int j = 0; j < 4 * N; j++) {
            y[j] = ynew[j];
        }

    }

    pos_x_file.close();
    pos_y_file.close();
    vel_x_file.close();
    vel_y_file.close();

    //Guardo condiciones iniciales
    ofstream cond_ini_file("resultados/cpp_cond_ini.txt");
    cond_ini_file << R0 << " " << v0 << " " << R0_dim << " " << v0_dim;


    return 0;
}