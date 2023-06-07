# Análisis 

Code Management (CD):
* Documentar funciones de cpp_cpu




Si cada paso es independiente de los demás, entonces puedo hacer estadística del mismo N usando los pasos de tiempo


También se calcularon el nro de interacciones por segundo que se pueden calcular

La charla tiene que ser de 15'. Mostrar motivación, cómo paralelizamos el código. Ver cómo escalea. Va a haber 4 fechas para presentar del 9 al 30 de junio. 



¿Qué falta hacer?






-Determinar qué N y dt analizar en el caso de python (el programa más exigente). Estos serán los que analizaré en los demás casos. No le debe tomar más de 20'. El dt tiene que ser suficientemente chico para que a tiempos largos no diverja el programa. PERO solo se ejecutará un único paso de tiempo en la versión final. En la corrida final, guardo_cada tiene que ser igual al nro de pasos totales, para guardar solo el dato final



-Ver cómo preguntar qué CPU y qué GPU me tocó. Esto estaría bueno agregarlo en la presentación final
-Ver cómo hacer profiling en el cluster del código py_cpu en un único paso de tiempo. Esto es solamente para la presentación.
-Hacer lo mismo para py_gpu
-Hacer lo mismo para cpp_cpu
-Hacer lo mismo para cpp_gpu_v1
-Hacer lo mismo para cpp_gpu_v2
-Analizar cada profiling y sacar bullet points de lo que está ocurriendo



-------------------------------
Tareas "boludas"
*Agregar el siguiente comentario a la última clase de CUDA:
En la primer versión cada iteración dura en torno a 3 s, mientras que con la segunda versión cada iteración dura en torno a 0.1 s. Luego, con la primer paralelización de CUDA tarda 0.01 s y con la segunda paralelización (usando memoria compartida) se llegó a 0.007 segundos.

















¿Cómo paralelizar mi código?
En 2 clases se va a explicar cómo hacer dinámica molecular con GPU. Creo que no me va a servir avanzar mucho y quizás sea mejor esperar a esa clase

*Hacer todo el trabajo que pueda en al GPU y solo copiar a la CPU los datos cada guardo_cada
*El nro de hilos por bloque debe ser mútliplo de 32
*Llenar el vector de condiciones iniciales en la GPU
*Puede ser útil guardar los datos en shared memory (en clase 4 diapositiva 14 se explica cómo hacerlo)
*De alguna manera debería "checkear" el resultado. Supongo que lo haré observando la conservación de la energía

-¿Qué estudiar?
*Tiempo de cómputo de py_cpu, cpp_cpu y cpp_gpu en función del nro de partículas. En algún momento la gpu debería serializar y aumentar linealmente el tiempo de cómputo. Estaría bueno
1. Hacer un único paso de tiempo y estudiar en función de nro de partículas, de modo de obtener una gráfica como la de la diapositiva 21 de la clase 2.
2. Hacer muchos pasos de tiempo y ver qué se obtiene
*Medir el speedup
*Hacer estadística sobre todo lo que mido!