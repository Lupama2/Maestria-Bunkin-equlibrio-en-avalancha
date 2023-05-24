# Análisis 

Code Management (CD):
* Documentar funciones de cpp_cpu


*Hacer f_maxwell() en cpp_gpu

#include <random>
#include <cmath>

double generateMaxwellBoltzmann(double T) {
    // Crear generadores de números aleatorios
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Generar tres números aleatorios con una distribución normal usando la transformación de Box-Muller
    double x1 = sqrt(-2.0 * log(distribution(generator))) * cos(2.0 * M_PI * distribution(generator));
    double x2 = sqrt(-2.0 * log(distribution(generator))) * sin(2.0 * M_PI * distribution(generator));
    double x3 = sqrt(-2.0 * log(distribution(generator))) * cos(2.0 * M_PI * distribution(generator));

    // Calcular la velocidad en función de la temperatura T
    double velocity = sqrt(x1 * x1 + x2 * x2 + x3 * x3) * sqrt(T);

    return velocity;
}



* Hacer cpp_cpu más eficiente:
-En avanzo_dt: Juntar la cuenta de distancia al origen adentro del for en el que se pregunta d_vec[i]>R0. Juntar este for con el siguiente. Intentar evitar declarar arrays.

No hago porque no sé qué efecto positivo puede llegar a tener:
-Evitar alocar memoria de dvx_vec y dvy_vec.

* Copiar cpp_cpu en cpp_gpu


¿Cómo paralelizar mi código?
En 2 clases se va a explicar cómo hacer dinámica molecular con GPU. Creo que no me va a servir avanzar mucho y quizás sea mejor esperar a esa clase


*Hacer todo el trabajo que pueda en al GPU y solo copiar a la CPU los datos cada guardo_cada
*El nro de hilos por bloque debe ser mútliplo de 32
*Llenar el vector de condiciones iniciales en la GPU
*Puede ser útil guardar los datos en shared memory (en clase 4 diapositiva 14 se explica cómo hacerlo)
*De alguna manera debería "checkear" el resultado. Supongo que lo haré observando la conservación de la energía
*Usar Thrust para poder compilar en CPU paralelo y poder testear en mi pc. Loc comandos de compilación están en la clase 6 diapositiva 3
*Los algoritmos de thrust se explican en clase 5. La 2da optimización hacerla con los algoritmos de la clase 6 (fusionando operaciones).
*Hago la operación matricial de py_cpu con thust?

¿Qué estudiar?
*Tiempo de cómputo de py_cpu, cpp_cpu y cpp_gpu en función del nro de partículas. En algún momento la gpu debería serializar y aumentar linealmente el tiempo de cómputo. Estaría bueno
1. Hacer un único paso de tiempo y estudiar en función de nro de partículas, de modo de obtener una gráfica como la de la diapositiva 21 de la clase 2.
2. Hacer muchos pasos de tiempo y ver qué se obtiene
*Medir el speedup
*Hacer estadística sobre todo lo que mido!