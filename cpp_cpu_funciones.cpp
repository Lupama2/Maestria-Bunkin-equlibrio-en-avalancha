// Importo librerías
#include <iostream>
#include <cmath>
#include <random>
//Para nros random:
#include <cstdlib>
// #include <ctime>

using namespace std;


const double pi = acos(-1.0);

void f(double* y, double* dydt, double alpha, int N) {
    /*
    Evaluación del miembro de la derecha del sistema de ecuaciones diferenciales del tipo dy_vec/dt = f(y_vec, t)
    Parameters
    ----------
    y_vec (dim 4N no modificado): vector de variables formado por [rx_vec, ry_vec, vx_vec, vy_vec] (en ese orden). Estos son:
        rx_vec (dim N): vector de posiciones en x de las partículas
        ry_vec (dim N): vector de posiciones en y de las partículas
        vx_vec (dim N): vector de velocidades en x de las partículas
        vy_vec (dim N): vector de velocidades en y de las partículas
    dydt (dim 4N modificado): vector de la derivada de variables.
    alpha: constante adimensional
    N: nro de partículas
   
    */
    double* rx_vec = y;
    double* ry_vec = &y[N];
    double* vx_vec = &y[2 * N];
    double* vy_vec = &y[3 * N];

    // Calculo drdt
    double* drx_vec = vx_vec;
    double* dry_vec = vy_vec;

    // Calculo dvdt
    double* dvx_vec = new double[N];
    double* dvy_vec = new double[N];
    for(int i = 0; i < N; ++i){
        dvx_vec[i] = 0;
        dvy_vec[i] = 0;
        for(int j = 0; j < N; ++j){
            if (i != j){
                double dx = rx_vec[i] - rx_vec[j];
                double dy = ry_vec[i] - ry_vec[j];
                double r = sqrt(dx*dx + dy*dy);
                double r3 = r*r*r;
                dvx_vec[i] += alpha*dx/r3;
                dvy_vec[i] += alpha*dy/r3;
            }
        }
    }

    // Concatenate results
    for (int i = 0; i < N; i++) {
        dydt[i] = drx_vec[i];
        dydt[N + i] = dry_vec[i];
        dydt[2 * N + i] = dvx_vec[i];
        dydt[3 * N + i] = dvy_vec[i];
    }


    // Clean up memory
    delete[] dvx_vec;
    delete[] dvy_vec;
}



double f_maxwell(){
    /*
    Rehacer. TO-DO.
    Por cómo se usa abajo, sólo genera 1 nro random
    */
    // random_device rd;
    // mt19937 gen(rd());
    // maxwell_distribution<double> dist(0.0, sqrt(m / (K * T0_dim)));
    // return dist(gen) * v0_dim;
    // Función para generar un número aleatorio en un rango específico
    return float(rand())/ float(RAND_MAX);

}

void condiciones_iniciales(double* y0, int N){
    /*
    Verificar funcionamiento de la generación de nros random. TO-DO
    */
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> rand_dist(0.0, 1.0);
    uniform_real_distribution<double> angle_dist(0.0, 2 * pi);

    //Lo siguiente es válido por cómo fue definida la adimnesionalización
    double R0 = 1.0;

    for (int i = 0; i < N; ++i){
        double r0 = R0 * rand_dist(gen);
        double tita0 = angle_dist(gen);
        y0[i] = r0 * cos(tita0); // = rx0_vec[i]
        y0[N + i] = r0 * sin(tita0); // = ry0_vec[i] 
    }

    for (int i = 0; i < N; ++i)
    {
        double v0 = f_maxwell();
        double tita0 = angle_dist(gen);
        y0[2 * N + i] = v0 * cos(tita0); // = vx0_vec[i]
        y0[3 * N + i] = v0 * sin(tita0); // = vy0_vec[i]
    }
}

void distancia_al_origen(double* r_vec, double* d_vec, int N){
    /*
    Verificar su funcionamiento. TO-DO
    Hacer Unit testing!
    */
    double* rx = &r_vec[0];
    double* ry = &r_vec[N];

    for (int i = 0; i < N; ++i){
        d_vec[i] = sqrt(rx[i] * rx[i] + ry[i] * ry[i]);
    }
}



void rebote_blando(double rx, double ry, double vx, double vy, double *result){

    result[0] = rx;
    result[1] = ry;

    double tita = atan2(ry, rx);
    result[2] = -vx * cos(2 * tita) - vy * sin(2 * tita);
    result[3] = -vx * sin(2 * tita) + vy * cos(2 * tita);

    return;
}


// Función para el método de Verlet
void metodoVerlet(double* yold, double t, double dt, int N, double* ynew, double alpha) {

    double* dydt = new double[4 * N]; // Array para almacenar dy/dt

    // Cálculo del vector de fuerzas
    double* F_vec = new double[2 * N];
    f(yold , dydt, alpha, N);
    //Copio los valores
    for (int i = 0; i < 2 * N; ++i){
        F_vec[i] = dydt[2 * N + i];
    }

    // Asignación del vector de posiciones
    // double* r_vec_new = new double[2 * N];

    // Cálculo de la posición en el siguiente paso de tiempo
    for (int i = 0; i < 2 * N; i++) {
        ynew[i] = yold[i] + yold[2 * N + i] * dt + 0.5 * dt * dt * F_vec[i];
    }

    // Cálculo de la fuerza en el siguiente paso de tiempo
    // double* ynew_partial = new double[4 * N];

    // Cálculo del vector de fuerzas
    double* F_vec_new = new double[2 * N];
    f(ynew, dydt, alpha, N);
    //Copio los valores
    for (int i = 0; i < 2 * N; ++i){
        F_vec_new[i] = dydt[2 * N + i];
    }

    // Cálculo de la velocidad en el siguiente paso de tiempo
    // double* v_vec_new = new double[2 * N];
    for (int i = 0; i < 2 * N; i++) {
        ynew[2*N + i] = yold[2 * N + i] + 0.5 * dt * (F_vec[i] + F_vec_new[i]);
    }

    //Asigno todo a ynew
    // for(int i = 0; i < N; ++i){
    //     ynew[2*N + i] = v_vec_new[i];
    //     ynew[3*N + i] = v_vec_new[N + i];
    // }

    // Liberación de memoria
    delete[] dydt;
    // delete[] r_vec_new;
    // delete[] v_vec_new;
}

// Función avanzo_dt
void avanzo_dt(double* y, double* ynew, double t, double dt, int N, double alpha) {
    double R0 = 1.;

    // double* ynew = new double[4 * N]; // Array para almacenar los nuevos valores de y

    // Avanzo un paso de tiempo
    metodoVerlet(y, t, dt, N, ynew, alpha);

    // Calculo la distancia de cada partícula al origen
    double* r_vec = ynew;
    double* d_vec = new double[N];
    distancia_al_origen(r_vec, d_vec, N);

    // Determino los índices en los que una partícula superó la distancia R0
    int* indices = new int[N];
    int numIndices = 0;
    for (int i = 0; i < N; i++) {
        if (d_vec[i] > R0) {
            indices[numIndices] = i;
            numIndices++;
        }
    }

    // Opero sobre las partículas que chocaron
    if (numIndices > 0) {
        for (int i = 0; i < numIndices; i++) {
            int indice = indices[i];

            // Obtengo las variables correspondientes
            double rx = ynew[indice];
            double ry = ynew[indice + N];
            double vx = ynew[indice + 2 * N];
            double vy = ynew[indice + 3 * N];

            // Rebote
            double* result = new double[4];
            rebote_blando(rx, ry, vx, vy, result);

            // Añado los nuevos datos en ynew de forma ordenada
            ynew[indice] = result[0];
            ynew[indice + N] = result[1];
            ynew[indice + 2 * N] = result[2];
            ynew[indice + 3 * N] = result[3];
        }
    }

    // Liberación de memoria
    delete[] d_vec;
    delete[] indices;
}
