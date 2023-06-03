// Importo librerías
#include <iostream>
#include <cmath>
#include <random>
//Para nros random:
#include <cstdlib>
// #include <ctime>

using namespace std;


const float pi = acos(-1.0);

void f(float* y, float* dydt, float alpha, int N) {
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
    float* rx_vec = y;
    float* ry_vec = &y[N];
    float* vx_vec = &y[2 * N];
    float* vy_vec = &y[3 * N];

    for(int i = 0; i < N; ++i){
        // Calculo drdt
        dydt[i] = vx_vec[i]; //drx_vec
        dydt[N + i] = vy_vec[i]; //dry_vec

        // Calculo dvdt
        //dvx_vec = dydt[2 * N + i]
        //dvy_vec = dydt[3 * N + i]
        dydt[2 * N + i] = 0;
        dydt[3 * N + i] = 0;
        for(int j = 0; j < N; ++j){
            if (i != j){
                float dx = rx_vec[i] - rx_vec[j];
                float dy = ry_vec[i] - ry_vec[j];
                float r = sqrt(dx*dx + dy*dy);
                float r3 = r*r*r;
                dydt[2 * N + i] += alpha*dx/r3;
                dydt[3 * N + i] += alpha*dy/r3;
            }
        }
    }


}





float f_maxwell_pura() {
    /*
    Los nros random de esta distribución siguen la distribución de maxwell-blotzmann

    f(v) = np.sqrt(2/pi)*v**2*exp(-v**2/2)

    Aú
    */


    // Crear generadores de números aleatorios con distribución normal
    static std::default_random_engine generator;
    static std::normal_distribution<float> distribution(0.0, 1.0);

    // Generar tres números aleatorios con una distribución normal usando la transformación de Box-Muller
    float x1 = distribution(generator);
    float x2 = distribution(generator);
    float x3 = distribution(generator);

    // Calcular la velocidad en función de la temperatura T
    float velocity = sqrt(x1 * x1 + x2 * x2 + x3 * x3);

    return velocity;
}

float f_maxwell_adim(){
    /*
    Falta corregir esta función
    
    */
    /*
    Rehacer. TO-DO.
    Por cómo se usa abajo, sólo genera 1 nro random
    */
    // random_device rd;
    // mt19937 gen(rd());
    // maxwell_distribution<float> dist(0.0, sqrt(m / (K * T0_dim)));
    // return dist(gen) * v0_dim;
    // Función para generar un número aleatorio en un rango específico

    return f_maxwell_pura()*sqrt(3);
}



void condiciones_iniciales(float* y0, int N){
    /*
    Verificar funcionamiento de la generación de nros random. TO-DO
    */
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> rand_dist(0.0, 1.0);
    uniform_real_distribution<float> angle_dist(0.0, 2 * pi);

    //Lo siguiente es válido por cómo fue definida la adimnesionalización
    float R0 = 1.0;

    for (int i = 0; i < N; ++i){
        float r0 = R0 * rand_dist(gen);
        float tita_r0 = angle_dist(gen);
        y0[i] = r0 * cos(tita_r0); // = rx0_vec[i]
        y0[N + i] = r0 * sin(tita_r0); // = ry0_vec[i] 

        float v0 = rand_dist(gen); //f_maxwell_adim();
        float tita_v0 = angle_dist(gen);
        y0[2 * N + i] = v0 * cos(tita_v0); // = vx0_vec[i]
        y0[3 * N + i] = v0 * sin(tita_v0); // = vy0_vec[i]
    }
}

// void distancia_al_origen(float* r_vec, float* d_vec, int N){
//     /*
//     Verificar su funcionamiento. TO-DO
//     Hacer Unit testing!
//     */
//     float* rx = &r_vec[0];
//     float* ry = &r_vec[N];

//     for (int i = 0; i < N; ++i){
//         d_vec[i] = sqrt(rx[i] * rx[i] + ry[i] * ry[i]);
//     }
// }

float distancia_al_origen(float rx, float ry){
    /*
    Verificar su funcionamiento. TO-DO
    Hacer Unit testing!
    */
   return sqrt(rx * rx + ry * ry);
}


void rebote_blando(float rx, float ry, float vx, float vy, float *vx_new, float* vy_new){

    // result[0] = rx;
    // result[1] = ry;

    float tita = atan2(ry, rx);
    *vx_new = -vx * cos(2 * tita) - vy * sin(2 * tita);
    *vy_new = -vx * sin(2 * tita) + vy * cos(2 * tita);

    return;
}

void correccion_Temperatura(float *vx_vec, float *vy_vec, int N){
    /*
    Verificar su funcionamiento. TO-DO
    Hacer Unit testing!
    */
    float denominador = 0;
    for (int i = 0; i < N; ++i){
        denominador += vx_vec[i] * vx_vec[i] + vy_vec[i] * vy_vec[i];
    }
    denominador = denominador;
    float lambda = sqrt((N) / denominador);
    for (int i = 0; i < N; ++i){
        vx_vec[i] = vx_vec[i] * lambda;
        vy_vec[i] = vy_vec[i] * lambda;
    }
    return;
}

// Función para el método de Verlet
void metodoVerlet(float* yold, float t, float dt, int N, float* ynew, float alpha) {

    float* dydt = new float[4 * N]; // Array para almacenar dy/dt
    float *dydt_new = new float[4 * N]; // Array para almacenar dy/dt en el siguiente paso de tiempo

    f(yold , dydt, alpha, N);

    //Asigno el vector de fuerzas
    float *F_vec = &dydt[2*N] ;

    // Cálculo de la posición en el siguiente paso de tiempo
    for (int i = 0; i < 2 * N; i++) {
        ynew[i] = yold[i] + yold[2 * N + i] * dt + 0.5 * dt * dt * F_vec[i];
    }

    // Cálculo de la fuerza en el siguiente paso de tiempo
    f(ynew, dydt_new, alpha, N);
    float *F_vec_new = &dydt_new[2*N] ;

    // Cálculo de la velocidad en el siguiente paso de tiempo
    // float* v_vec_new = new float[2 * N];
    for (int i = 0; i < 2 * N; i++) {
        ynew[2*N + i] = yold[2 * N + i] + 0.5 * dt * (F_vec[i] + F_vec_new[i]);
    }

    // Liberación de memoria
    delete[] dydt;
    delete[] dydt_new;
}

// Función avanzo_dt
void avanzo_dt(float* y, float* ynew, float t, float dt, int N, float alpha) {
    float R0 = 1.;

    // float* ynew = new float[4 * N]; // Array para almacenar los nuevos valores de y

    // Avanzo un paso de tiempo
    metodoVerlet(y, t, dt, N, ynew, alpha);

    // Calculo la distancia de cada partícula al origen
    float* r_vec = ynew;

    // Determino los índices en los que una partícula superó la distancia R0
    for (int i = 0; i < N; i++) {
        //Opero sobre las partículas que chocaron
        if (distancia_al_origen(r_vec[i], r_vec[i + N]) > R0) {

            // Obtengo las variables correspondientes
            float rx = ynew[i];
            float ry = ynew[i + N];
            float vx = ynew[i + 2 * N];
            float vy = ynew[i + 3 * N];

            // Rebote
            float* result = new float[2];
            rebote_blando(rx, ry, vx, vy, &ynew[i + 2 * N], &ynew[i + 3 * N]);

            // Añado los nuevos datos en ynew de forma ordenada
            ynew[i] = rx;
            ynew[i + N] = ry;
            // ynew[indice + 2 * N] = result[0];
            // ynew[indice + 3 * N] = result[1];

        }

    }
    // Reescaleo la temperatura
    correccion_Temperatura(&ynew[2*N], &ynew[3*N], N);

}
