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


¿Qué falta hacer?
-Verificar el funcionamiento de f_maxwell(): generar N nros con la distribución uniforme y ver que lo sea. generar N nros con la distribución de maxwell y ver que lo sea
-Determinar qué N y dt analizar en el caso de python (el programa más exigente). Estos serán los que analizaré en los demás casos. No le debe tomar más de 20'. El dt tiene que ser suficientemente chico para que a tiempos largos no diverja el programa. PERO solo se ejecutará un único paso de tiempo en la versión final.
-¿Podría recibir N y dt desde la cmd como los cpp de clases? En caso negativo, reducir el programa de python a una función, de modo de poder poner de input varios N y dt, para luego correr en el cluster.
-Incluir la repetición M de cada caso para tener estadística
-Que tome los tiempos y los guarde en un archivo. En caso de que pueda recibir N y dt desde la cmd, imprimir directamente los tiempos. Luego configuraré el comando para que guarde los outputs en un archivo
-Ver cómo hacer profiling en el cluster del código py en un único paso de tiempo. Esto es solamente para la presentación.
-Ver cómo hacer profiling en el cluster del código cpp en un único paso de tiempo. Esto es solamente para la presentación.

-Hacer la versión paralela de py: cambian numpy por cupy Y NADA MÁS

-Preparar el archivo de cpp_cpu para que tome N y dt de consola
-Ver qué línea debería correr desde el cluster para hacer un loop sobre N y dt.

-Preparar en graficos.ipynb gráficos de tiempo de cómputo de cada programa
-Graficar el tiempo de cómputo para N fijo y un único paso de tiempo y distintos dt. No debería depender de dt
-Graficar speed-up de
.py_gpu respecto a py_cpu
.cpp_cpu respecto a py_cpu
-cpp_gpu respecto a py_cpu
.cpp_gpu respecto a cpp_cpu

-Documentar funciones en cpp_cpu
-Ver cómo preguntar qué CPU y qué GPU me tocó. Esto estaría bueno agregarlo en la presentación final

-Instalar CUDA toolkint. Correr un programa de ejemplo de thrust. Pedírselo a ChatGPT

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

-¿Qué estudiar?
*Tiempo de cómputo de py_cpu, cpp_cpu y cpp_gpu en función del nro de partículas. En algún momento la gpu debería serializar y aumentar linealmente el tiempo de cómputo. Estaría bueno
1. Hacer un único paso de tiempo y estudiar en función de nro de partículas, de modo de obtener una gráfica como la de la diapositiva 21 de la clase 2.
2. Hacer muchos pasos de tiempo y ver qué se obtiene
*Medir el speedup
*Hacer estadística sobre todo lo que mido!