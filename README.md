# Maestria-Bunkin-equlibrio-en-avalancha
 Análisis del equilibrio de un gas de electrones en un recinto circular

Inserto un emoji para que se vea en el cluster


Poner el pdf de la presentación en el cluster tmb
5NElCjSEafnjsaxT9r5C//

## Introducción
 El objetivo del proyecto es simular la dinámica de un gas de N electrones contenido en un recinto circular de radio $R0$. Para ello se cuenta con las ecuaciones de movimiento dadas por la ley de Newton y la ley de Lorentz, adimensionalizadas. Como condición inicial se parte de N electrones en posiciones aleatorias y velocidades dadas por una distribución de Maxwell-Boltzmann a temperatura T0 = 1000 K.

 Se asume que las colisiones con la pared son elásticas, considerando la misma como una "pared blanda". Esto significa que durante la evolución la partícula invierte su velocidad si atraviesa la pared, pero no se refleja su posición. Esto puede ser útil para evitar discontinuidades en la energía total del sistema.

 Además, el sistema se evoluciona mediante el método de Verlet


## Definición del problema


CI
Ecuaciones diferenciales
Partículas en una caja


El objetivo del proyecto es simular la dinámica de un gas de N electrones contenido en un recinto circular de radio $R0$. Para ello se cuenta con las ecuaciones de movimiento dadas por la ley de Newton y la ley de Lorentz, adimensionalizadas. Como condición inicial se parte de N electrones en posiciones aleatorias y velocidades dadas por una distribución de Maxwell-Boltzmann a temperatura T0 = 1000 K.

Se asume que las colisiones con la pared son elásticas, considerando la misma como una "pared blanda". Esto significa que durante la evolución la partícula invierte su velocidad si atraviesa la pared, pero no se refleja su posición. Esto puede ser útil para evitar discontinuidades en la energía total del sistema.

Además, el sistema se evoluciona mediante el método de Verlet




## Motivación
 El estudio de las propiedades en el equilibrio podría permitir estudiar el proceso de evolución de una avalancha de electrones generada en una cavidad rodeada por una fuente de electrones.








## Implementación

*¿Qué cálculos se realizan en el código? En definitiva, son los nombres de los kernels de cpp_gpu


*¿Cómo se resuelve el problema?
-Adimensionalización
-CI aleatorias para posición y velocidades
-Evolución mediante el método de Verlet (método simpléctico? conservativo)
-Condición de pared blanda (gráfico)
-Corrección de T

## Versiones en serie y en paralelo
*Contar cada versión por separado. Hacer una tabla en la que diga
-Lenguaje
-Descripción de la paralelización
Versión 1	Python	En serie (numpy)
Versión 2	Python	En paralelo (numpy -> cupy)
Versión 3	C++	En serie
Versión 4	CUDA C	En paralelo (kernels)
Versión 5	CUDA C	En paralelo (kernels + shared memory)
Que lo anterior funcione como resumen. Me gustaría hacer una "historia" de cómo fui cambiando de una versión a la otra y en el camino menciono cómo fui haciendo las cuentas. Esto debería intercalarse con el profiling y los gráficos de speed-up
Mencionar:
*En la versión 1 calculé las fuerzas matricialmente, de modo de mantenerme dentro de numpy
*En la versión 2 sólo intercambié numpy por cupy, como para demostrar qué speed-up uno podría obtener con un código intercambiando solo una línea
*En la versión 3 usé los loops de C++
*En la versión 4 usé kernels para hacer cada una de las cuentas, un blocksize de ... y solo hice copias al inicio y al final
*En la versión 5 usé shared-memory
-Animación o gráfico de los resultados
