# Maestria-Bunkin-equlibrio-en-avalancha
 Análisis del equilibrio de un gas de electrones en un recinto circular

5NElCjSEafnjsaxT9r5C//

## Introducción
 El objetivo del proyecto es simular la dinámica de un gas de N electrones contenido en un recinto circular de radio $R_0$. Para ello se cuenta con las ecuaciones de movimiento dadas por la ley de Newton y la ley de Lorentz, adimensionalizadas. Como condición inicial se parte de N electrones en posiciones aleatorias y velocidades dadas por una distribución de Maxwell-Boltzmann a temperatura $T_0 = 1000$ K.

 Se asume que las colisiones con la pared son elásticas, considerando la misma como una "pared blanda". Esto significa que durante la evolución la partícula invierte su velocidad si atraviesa la pared, pero no se refleja su posición. Esto puede ser útil para evitar discontinuidades en la energía total del sistema.

 Además, el sistema se evoluciona mediante el método de Verlet

## Motivación
 El estudio de las propiedades en el equilibrio podría permitir estudiar el proceso de evolución de una avalancha de electrones generada en una cavidad rodeada por una fuente de electrones.

## Implementación

*¿Qué cálculos se realizan en el código?
-Se establecen las Condiciones Iniciales (CI) aleatorias para posición y velocidades
-Se evolucionan las partículas mediante el método de Verlet
-Durante la evolución se controla si alguna partícula atravesó la pared, en caso positivo se invierte su velocidad normal
-En cada paso de tiempo se calcula la temperatura del sistema y luego se corrije para alcanzar la temperatura de equilibrio deseada.

## Versiones en serie y en paralelo
Existen 5 versiones del código, 2 de las cuales son en serie y 3 en paralelo.
| Versión   | Lenguaje | job            | Comentarios                                |
|-----------|----------|----------------|--------------------------------------------|
| Versión 1 | Python   | job_py_cpu     | En serie (numpy)                           |
| Versión 2 | Python   | job_py_gpu     | En paralelo (numpy -> cupy)                |
| Versión 3 | C++      | job_cpp_cpu    | En serie                                   |
| Versión 4 | CUDA C++ | job_cpp_gpu_v1 | En paralelo (kernels)                      |
| Versión 5 | CUDA C++ | job_cpp_gpu_v2 | En paralelo (kernels + shared memory)      |
Existe un 6to job llamado job_cpp_gpu_evolucion que permite calcular la evolución hasta el equilibrio y guardar cada pocos pasos de tiempo empleando la versión 4. Para ejecutar cada una de las versiones, simplemente hacer
~~~
qsub job_version
~~~
en el cluster y se estará compilando el código correspondiente y ejecutando el código

Se puede graficar la solución en graficos_evolucion y el rendimiendo de cada código respecto a los demás con graficos_rendimiento. En este último es importante haber ejecutado todos los jobs para que se grafique correctamente.



