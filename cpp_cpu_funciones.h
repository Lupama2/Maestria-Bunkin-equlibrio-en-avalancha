
void f(float* y, float* dydt, float alpha, int N);
float f_maxwell_adim();
float f_maxwell_pura();
void condiciones_iniciales(float* y0, int N);
void distancia_al_origen(float* r_vec, float* d_vec, int N);
void rebote_blando(float rx, float ry, float vx, float vy, float *result);
void metodoVerlet(float* yold, float t, float dt, int N, float* ynew, float alpha);
void avanzo_dt(float* y, float* ynew, float t, float dt, int N, float alpha);
