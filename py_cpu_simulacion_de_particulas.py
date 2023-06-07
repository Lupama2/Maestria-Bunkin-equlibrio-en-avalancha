#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Lupama2/Maestria-Bunkin-equlibrio-en-avalancha/blob/dev/py_cpu_simulacion_de_particulas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# # Simulación de partículas
# 
# Versión en Python. Siempre que fue posible se empleó numpy para reemplazar los lazos for. De esta manera, el cálculo de las fuerzas se realiza matricialmente

# ## Importo dependencias

# In[1]:


#Importo librerías
import numpy as np
import os
import time
import sys


# ## Describo características del hardware
# 
# Esta sección del código no se puede ejecutar en el cluster

# In[6]:


# import cpuinfo

# cpu_info = cpuinfo.get_cpu_info()

# print("Información de la CPU:")
# print("Nombre:", cpu_info['brand_raw'])
# print("Arquitectura:", cpu_info['arch'])
# print("Frecuencia:", cpu_info['hz_actual'])
# print("Núcleos físicos:", cpu_info['count'])


# ## Defino ctes

# In[ ]:


#Constantes matemáticas
pi = np.pi

#Ctes físicas
m = 9.11e-31*1e3 #[g]
e = 1.602e-19*(1/3.336641e-10) #[Fr]
c = 299792458*1e2 #[cm/s]
K = 1.380649e-23*(1/1e-7) #constante de Boltzmann [ergio/K], obtenida de NIST


# ## C.I. (Condiciones Iniciales)
# 
# A partir de ahora todas las variables definidas son adimensionales. Las variables que posean dimensiones se declararán con el sufico _dim

# In[ ]:


#Radio del círculo y velocidad inicial de la partícula
R0_dim = 1e-6 #[cm]
T0_dim = 1000 #[K]


# In[ ]:


#Calculo las ctes adimensionales
v0_dim = (2*K*T0_dim/m)**(1/2)
alpha = e**2/m/R0_dim/(v0_dim**2)
print(f"Constante adimensional, alpha = {alpha}")

#Radio del círculo y velocidad inicial de la partícula adimensionales
R0 = 1
v0 = 1


# ## Sistema de ecuaciones diferenciales

# In[ ]:


def f(y, N):
    '''
    Evaluación del miembro de la derecha del sistema de ecuaciones diferenciales del tipo dy_vec/dt = f(y_vec, t)
    Parameters
    ----------
    y_vec (ndarray de dimensión 4N): vector de variables formado por [rx_vec, ry_vec, vx_vec, vy_vec] (en ese orden). Estos son:
        rx_vec (ndarray de dimensión N): vector de posiciones en x de las partículas
        ry_vec (ndarray de dimensión N): vector de posiciones en y de las partículas
        vx_vec (ndarray de dimensión N): vector de velocidades en x de las partículas
        vy_vec (ndarray de dimensión N): vector de velocidades en y de las partículas
    N (int): nro de partículas
    
    '''
    rx_vec = y[:N]
    ry_vec = y[N:2*N]
    vx_vec = y[2*N:3*N]
    vy_vec = y[3*N:4*N]


    #Calculo drdt
    drx_vec = vx_vec
    dry_vec = vy_vec

    #Calculo dvdt
    Ax_matriz = np.tile(rx_vec, (N,1))
    Ay_matriz = np.tile(ry_vec, (N,1))

    Bx_matriz = Ax_matriz.T - Ax_matriz
    By_matriz = Ay_matriz.T - Ay_matriz

    modulo = np.sqrt(Bx_matriz**2 + By_matriz**2)
    modulo = np.where(modulo == 0, 1, modulo) #Para evitar división por cero

    Cx_matriz = Bx_matriz/modulo**3
    Cy_matriz = By_matriz/modulo**3

    dvx_vec = alpha*np.sum(Cx_matriz, axis=1)
    dvy_vec = alpha*np.sum(Cy_matriz, axis=1)
   
    dydt = np.concatenate((drx_vec, dry_vec, dvx_vec, dvy_vec))

    return dydt


# In[ ]:


def condiciones_iniciales(N):
    '''
    Devuelve un vector de condiciones iniciales adimensionales

    Parameters
    ----------
    N (int): nro de partículas

    Nota:
    (1) Las posiciones iniciales se distribuyen aleatoriamente en el círculo de radio R0
    (2) Las velocidades iniciales se distribuyen de acuerdo a una distribución de Maxwell-Boltzmann, pero su dirección es aleatoria con distribución uniforme
    '''
    y0 = np.empty(4*N)

    #Posiciones aleatorias
    r0_vec = R0*np.random.rand(N)
    tita0_vec = 2*pi*np.random.rand(N)
    y0[:N] = r0_vec*np.cos(tita0_vec) # = rx0_vec[i]
    y0[N:2*N] = r0_vec*np.sin(tita0_vec) # = ry0_vec[i] 

    #Velocidades de acuerdo a una distribución de Maxwell-Boltzmann 
    v0_vec = v0*np.random.rand(N)
    tita0_vec = 2*pi*np.random.rand(N)
    y0[2*N:3*N] = v0_vec*np.cos(tita0_vec) # = vx0_vec[i]
    y0[3*N:4*N] = v0_vec*np.sin(tita0_vec) # = vy0_vec[i]

    return y0


# In[ ]:


def distancia_al_origen(r_vec, N):
    '''
    Calcula la distancia al origen de las partículas.

    Parameters
    ----------
    r_vec (ndarray de dimensión 2N): vector de posiciones de las partículas tal que r_vec = [rx_vec, ry_vec]
    N (int): nro de partículas
    '''

    #Opero de forma similar a como lo hice dentro de f
    rx = r_vec[:N]
    ry = r_vec[N:]

    d_vec = np.sqrt(rx**2 + ry**2)

    return d_vec

def correccion_Temperatura(vx_vec, vy_vec):
    '''
    Corrección de las velocidades para que la temperatura sea la deseada (T0)

    Parameters
    ----------
    vx_vec (ndarray de dimensión N): vector de velocidades en x de las partículas
    vy_vec (ndarray de dimensión N): vector de velocidades en y de las partículas

    Returns
    -------
    vx_vec_new (ndarray de dimensión N): vector de velocidades en x de las partículas corregidas
    vy_vec_new (ndarray de dimensión N): vector de velocidades en y de las partículas corregidas
    '''
    #Cuentas previas
    N = len(vx_vec)
    #Calculo el factor de corrección
    lambda_ = np.sqrt((N)/(np.sum(vx_vec**2 + vy_vec**2)))
    
    #Corrijo
    vx_vec_new, vy_vec_new = lambda_*vx_vec, lambda_*vy_vec

    return vx_vec_new, vy_vec_new


# ## Funciones útiles para la evolución

# In[ ]:


def rebote_blando(rx, ry, vx, vy):
    '''
    Calcula las posiciones y velocidades de una partícula luego del rebote con la pared. Se considera pared blanda, es decir, en el choque solo se invierte la velocidad.

    Parameters
    ----------
    rx (float): posición x de la partícula
    ry (float): posición y de la partícula
    vx (float): velocidad x de la partícula
    vy (float): velocidad y de la partícula

    Returns
    -------
    rx_new (float): nueva posición x de la partícula
    ry_new (float): nueva posición y de la partícula
    vx_new (float): nueva velocidad x de la partícula
    vy_new (float): nueva velocidad y de la partícula
    '''

    #Calculo la nueva posición
    rx_new = rx
    ry_new = ry

    #Calculo la nueva velocidad
    
    #Calculo el ángulo del vector [rx,ry]
    tita = np.arctan2(ry, rx)
    vx_new = -vx*np.cos(2*tita) - vy*np.sin(2*tita)
    vy_new = -vx*np.sin(2*tita) + vy*np.cos(2*tita)

    return rx_new, ry_new, vx_new, vy_new

def metodo_Verlet(yold, t, dt, N):
    '''
    Método de Verlet

    Parameters
    ----------
    yold (ndarray de dimensión 4N): vector de posiciones y velocidades de las partículas en el paso de tiempo anterior
    t (float): tiempo en el paso de tiempo anterior
    dt (float): paso temporal
    N (int): nro de partículas

    Returns
    -------
    ynew (ndarray de dimensión 4N): vector de posiciones y velocidades de las partículas en el paso de tiempo siguiente

    Nota: el esquema numérico está bien definido en el apunte de Física Computacional (FISCOM)
    '''
    
    #Calculo el vector de fuerzas
    dydt = f(yold, N) 
    F_vec = dydt[2*N:]

    #Asigno el vector de posiciones
    r_vec = yold[:2*N]
    #Calculo la posición en el siguiente paso de tiempo
    r_vec_new = r_vec + yold[2*N:4*N]*dt + 1/2*dt**2*F_vec

    #Calculo la fuerza en el siguiente paso de tiempo
    ynew_partial = np.concatenate((r_vec_new, yold[2*N:])) #corrijo solo r_vec

    dydt_new = f(ynew_partial, N)
    F_vec_new = dydt_new[2*N:]

    #Calculo la velocidad en el siguiente paso de tiempo
    v_vec_new = yold[2*N:] + 1/2*dt*(F_vec + F_vec_new)

    return np.concatenate((r_vec_new, v_vec_new))


def avanzo_dt(y, t, dt, N, metodo):
    '''
    Avanza un paso de tiempo el sistema

    Parameters
    ----------
    y (ndarray de dimensión 4N): vector de posiciones y velocidades de las partículas en el paso anterior
    t (float): tiempo en el paso anterior
    dt (float): paso temporal
    N (int): nro de partículas
    metodo (función): método de integración a utilizar

    Returns
    -------
    ynew (ndarray de dimensión 4N): vector de posiciones y velocidades de las partículas en el paso siguiente
    '''
    #Avanzo un paso de tiempo
    ynew = metodo(y, t, dt, N)

    #Verifico si se cumple la condición de choque
    #Calculo la distancia de cada partícula al origen
    d_vec = distancia_al_origen(ynew[0:2*N], N)

    #Determino todos los índices en los que una partícula superó la distancia R0
    indices = np.where(d_vec>R0)[0]
    #Opero sobre las partículas que chocaron
    if len(indices) > 0:
        for indice in indices:
            #Calculo las variables correspondientes
            rx = ynew[indice]
            ry = ynew[indice+N]
            vx = ynew[indice+2*N]
            vy = ynew[indice+3*N]

            #Rebota
            rx_new, ry_new, vx_new, vy_new = rebote_blando(rx, ry, vx, vy)
            #Añado los nuevos datos en ynew de forma ordenada
            ynew[indice] = rx_new
            ynew[indice+N] = ry_new
            ynew[indice+2*N] = vx_new
            ynew[indice+3*N] = vy_new

    #Corrijo las velocidades para llegar a T0. Al hacer esta corrección se deja de conservar la energía cinética
    ynew[2*N:3*N], ynew[3*N:4*N] = correccion_Temperatura(ynew[2*N:3*N], ynew[3*N:4*N])

    return ynew


# In[ ]:


def savetxt_append(filename, array, append = True):
    '''
    Guarda array en filename en formato txt. Se puede hacer append de los datos o no

    Parameters
    ----------
    filename (str): nombre del archivo con terminación ".txt"
    array (ndarray): array a guardar
    append (bool): si se quiere hacer append de los datos o no   
    '''
    if append == True:
        with open(filename, "ab") as f:
            # f.write(b"\n")
            #Save array as row
            np.savetxt(f, [array], delimiter = "\t")
    else:
        with open(filename, "wb") as f:
            # f.write(b"\n")
            np.savetxt(f, [array], delimiter = "\t")


# ## Solución numérica

# ### Parámetros de la evolución

# In[ ]:


N= 1000 #10 #Nro de partículas
dt = 1e-5 #En 1e-1 ya genera problemas. Las partículas salen de la circunferencia. A modo gral, a mayor N, menor dt
n_pasos = 10 #20000*5*10
guardo_cada = 100 #Cada cuántos pasos de tiempo se guardarán los datos

if not('ipykernel' in sys.modules):
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
        dt = float(sys.argv[2])
        n_pasos = int(sys.argv[3])
        guardo_cada = int(sys.argv[4])

print(f"N = {N}")


# In[ ]:



#Condiciones iniciales
y0 = condiciones_iniciales(N)
y = y0.copy()
t = 0

#Nombres de archivos
files = ["resultados/py_cpu_pos_x.txt", "resultados/py_cpu_pos_y.txt", "resultados/py_cpu_vel_x.txt", "resultados/py_cpu_vel_y.txt"]

t_computo_total = 0
t_computo_pasos = np.empty(n_pasos)
for i in range(n_pasos):
    # Comienzo a medir el tiempo
    t_computo_inicial = time.time()

    if i%guardo_cada== 0:
        print(f"t = {round(t,2)}\tEvolución al {round(i/n_pasos*100,2)}%")

    t += dt
    if i % guardo_cada == 0:

        #Guardo datos
        for j in range(4):
            #Guardar array separado por comma
            savetxt_append(files[j], y[j*N:(j+1)*N], False if i == 0 else True)
    y = avanzo_dt(y, t, dt, N, metodo_Verlet)

    #Termino de medir el tiempo
    t_computo_final = time.time()
    t_computo_pasos[i] = t_computo_final - t_computo_inicial


# In[ ]:


# Continúo la evolución por n2_pasos pasos de tiempo


# n2_pasos = 2*n_pasos #Correspondientes al segundo ciclo de evolución

# pos_x = np.concatenate((pos_x, np.empty([n2_pasos//guardo_cada, N_esperado])))
# pos_y = np.concatenate((pos_y, np.empty([n2_pasos//guardo_cada, N_esperado])))
# vel_x = np.concatenate((vel_x, np.empty([n2_pasos//guardo_cada, N_esperado])))
# vel_y = np.concatenate((vel_y, np.empty([n2_pasos//guardo_cada, N_esperado])))
# q_tot = np.concatenate((q_tot, np.zeros([n2_pasos//guardo_cada, N_esperado])))


# for i in range(n_pasos, n_pasos + n2_pasos):
#     try:
#         y = avanzo_dt(y, q_vec, t, dt, metodo)
#     except ValueError:
#         print(f"Último índice: {i}")
#         break

#     if i%guardo_cada == 0:
#         print(f"t = {round(t,2)}\tEvolución al {round(i/(n_pasos+n2_pasos)*100,2)}%\tN = {np.sum(q_vec)}")


#     t += dt
#     if i%guardo_cada == 0:
#         pos_x[i//guardo_cada] = np.concatenate((y[0:len(y)//4], np.zeros( N_esperado - len(y)//4)))
#         pos_y[i//guardo_cada] = np.concatenate((y[len(y)//4:2*len(y)//4], np.zeros( N_esperado - len(y)//4)))
#         vel_x[i//guardo_cada] = np.concatenate((y[2*len(y)//4:3*len(y)//4], np.zeros( N_esperado - len(y)//4)))
#         vel_y[i//guardo_cada] = np.concatenate((y[3*len(y)//4:4*len(y)//4], np.zeros( N_esperado - len(y)//4)))
#         q_tot[i//guardo_cada] = q_vec

# n_pasos = n2_pasos + n_pasos

# print("N_tot = ", np.sum(q_vec))


# In[ ]:


#Exporto los tiempos
savetxt_append("resultados/py_cpu_t_computo.txt", np.concatenate((np.array([N]), t_computo_pasos)), True)
# savetxt_append("resultados/py_cpu_t.txt", np.array([t, dt, n_pasos, guardo_cada]), False)
savetxt_append("resultados/py_cpu_t.txt", np.arange(0, dt*n_pasos, guardo_cada*dt), False)
#Exporto las condiciones iniciales
savetxt_append("resultados/py_cpu_cond_ini.txt", np.array([R0, v0, R0_dim, v0_dim]), False)

