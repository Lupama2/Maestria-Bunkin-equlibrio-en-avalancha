# Análisis 

Code Management (CD):
* Documentar funciones de cpp_cpu




Si cada paso es independiente de los demás, entonces puedo hacer estadística del mismo N usando los pasos de tiempo

En la primer versión cada iteración dura en torno a 3 s, mientras que con la segunda versión cada iteración dura en torno a 0.1 s. Luego, con la primer paralelización de CUDA tarda 0.01 s y con la segunda paralelización (usando memoria compartida) se llegó a 0.007 segundos.

También se calcularon el nro de interacciones por segundo que se pueden calcular

La charla tiene que ser de 15'. Mostrar motivación, cómo paralelizamos el código. Ver cómo escalea. Va a haber 4 fechas para presentar del 9 al 30 de junio. 



¿Qué falta hacer?
-¿Podría recibir N y dt desde la cmd como los cpp de clases? En caso negativo, reducir el programa de python a una función, de modo de poder poner de input varios N y dt, para luego correr en el cluster.
-Determinar qué N y dt analizar en el caso de python (el programa más exigente). Estos serán los que analizaré en los demás casos. No le debe tomar más de 20'. El dt tiene que ser suficientemente chico para que a tiempos largos no diverja el programa. PERO solo se ejecutará un único paso de tiempo en la versión final.
-Que tome los tiempos y los guarde en un archivo. En caso de que pueda recibir N y dt desde la cmd, imprimir directamente los tiempos. Luego configuraré el comando para que guarde los outputs en un archivo

-Hacer la versión paralela de py: cambian numpy por cupy Y NADA MÁS

-Ver cómo hacer profiling en el cluster del código py en un único paso de tiempo. Esto es solamente para la presentación.
-Ver cómo hacer profiling en el cluster del código cpp en un único paso de tiempo. Esto es solamente para la presentación.


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

GPT4 no puede hacerme todo, pero sí darme una mano. Le di el siguiente mensaje, no me supo paralelizar todo el código:
Estoy simulando un gas de electrones interactuantes contenidos en un círculo de radio constante en el tiempo. Los electrones tienen colisiones elásticas con la pared. Los electrones interactúan entre sí mediante la fuerza de coulomb. Tengo un código en C++ que calcula la evolución del gas de N electrones considerándolos como partículas individuales. Quiero paralelizarlo usando thrust. ¿Me puedes dar una mano? El código es relativamente largo


-Describir cómo quiero paralelizar mi código. Rearmar el código en serie con el paradigma en paralelo. Así será más fácil usar thrust luego (y GPT4).
. condiciones_iniciales implica llenar un vector. Creo que se puede hacer enteramente en paralelo
. El cálculo de f en py era llenar una matriz y luego hacer una reducción. Creo que tmb se puede paralelizar bajo ese concepto
. metodoVerlet se podría hacer enteramente en device. Los fors loops de asignación de ynew se podrían intercambiar por transform
. En avanzo_dt, el for se puede reemplazar por un kernel. Para hacer esto debería sacar la función rebote_blando.



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