
void f(double* y, double* dydt, double alpha, int N);
double f_maxwell();
void condiciones_iniciales(double* y0, int N);
void distancia_al_origen(double* r_vec, double* d_vec, int N);
void rebote_blando(double rx, double ry, double vx, double vy, double *result);
void metodoVerlet(double* yold, double t, double dt, int N, double* ynew, double alpha);
void avanzo_dt(double* y, double* ynew, double t, double dt, int N, double alpha);
