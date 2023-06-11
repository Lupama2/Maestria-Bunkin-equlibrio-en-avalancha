
/*
Para compilar
g++ -O3 cpp_cpu_funciones.cpp cpp_cpu_simulacion_de_particulas.cpp -o cpp_cpu_simulacion_de_particulas


*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include "cpp_cpu_funciones.h"
#include "timer.h"


using namespace std;

//Constantes matemáticas
const float pi = acos(-1.0);

// Constantes físicas
const float m = 9.11e-31 * 1e3; // [g]
const float e = 1.602e-19 * (1 / 3.336641e-10); // [Fr]
const float c = 299792458 * 1e2; // [cm/s]
const float K = 1.380649e-23 * (1 / 1e-7); // constante de Boltzmann [ergio/K], obtenida de NIST

// Radio del círculo y velocidad inicial de la partícula
const float R0_dim = 1e-6; // [cm]
const float T0_dim = 1000; // [K]



// srand(time(nullptr)); // Inicializar la semilla aleatoria con el tiempo actual



int main(const int argc, const char** argv) {

    // Nro de partículas
    int N = 1000; 
    float dt =  1e-5; //1e-8;
    int n_pasos = 10;
    int guardo_cada = 10;  // Valor deseado para guardo_cada

    if (argc > 1){
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        n_pasos = atoi(argv[3]);
        guardo_cada = atoi(argv[4]);};

    cout << "N = " << N << " particulas" << endl;
    cout << "dt = " << dt << endl;
    cout << "n_pasos = " << n_pasos << endl;
    cout << "guardo_cada = " << guardo_cada << endl;
    




    // Cálculo de las constantes adimensionales
    float v0_dim = sqrt(2 * K * T0_dim / m);
    float alpha = pow(e, 2) / (m * R0_dim * pow(v0_dim, 2));
    cout << "Constante adimensional, alpha = " << alpha << endl;

    // Radio del círculo y velocidad inicial de la partícula adimensionales
    float R0 = 1;
    float v0 = 1;

    cout << "Radio y velocidad iniciales adimensionales: " << R0 << ",\t" << v0 << endl;

    //Declaro array de datos
    float y[4 * N];
    float ynew[4 * N];

    // Condiciones iniciales
    srand(time(nullptr));
    condiciones_iniciales(y, N);

    float t = 0;

    ofstream pos_x_file("resultados/cpp_cpu_pos_x.txt");
    ofstream pos_y_file("resultados/cpp_cpu_pos_y.txt");
    ofstream vel_x_file("resultados/cpp_cpu_vel_x.txt");
    ofstream vel_y_file("resultados/cpp_cpu_vel_y.txt");
    ofstream t_file("resultados/cpp_cpu_t.txt");
    ofstream t_computo_file("resultados/cpp_cpu_t_computo.txt", ios_base::app);
    
    cout << "Archivos creados correctamente" << endl;
    float t_computo_total = 0.0;

    for (int i = 0; i < n_pasos; i++) {

        StartTimer();

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


        const float t_computo_paso = GetTimer() / 1000.0;
        //No tiro ninguna iteración
        t_computo_total += t_computo_paso;
        if (i == 0){
            t_computo_file << N << " ";
        }
        t_computo_file << t_computo_paso << " ";
        if (i == n_pasos - 1){
            t_computo_file << "\n";
        }
    
    }

    pos_x_file.close();
    pos_y_file.close();
    vel_x_file.close();
    vel_y_file.close();
    t_file.close();
    t_computo_file.close();

    //Guardo condiciones iniciales
    ofstream cond_ini_file("resultados/cpp_cpu_cond_ini.txt");
    cond_ini_file << R0 << " " << v0 << " " << R0_dim << " " << v0_dim;


    return 0;
}
