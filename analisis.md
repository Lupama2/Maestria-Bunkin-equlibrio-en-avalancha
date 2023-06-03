# Análisis 

Code Management (CD):
* Documentar funciones de cpp_cpu




Si cada paso es independiente de los demás, entonces puedo hacer estadística del mismo N usando los pasos de tiempo


También se calcularon el nro de interacciones por segundo que se pueden calcular

La charla tiene que ser de 15'. Mostrar motivación, cómo paralelizamos el código. Ver cómo escalea. Va a haber 4 fechas para presentar del 9 al 30 de junio. 



¿Qué falta hacer?























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