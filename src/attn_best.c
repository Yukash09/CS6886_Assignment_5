/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
 *
 * Placeholder: replace the body of attention() with your implementation.
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#define R 8
#define C 8
#define BR 64
#define BC 256

static inline float fast_exp(float x)
{
    if(x < -87.0f) x = -87.0f ;
    float n = floorf(x*1.44269504f + 0.5f) ;
    float r = x - n*0.693359375f + n*2.12194440e-4f ;
    float p = 1.9875691500e-4f ;
    p = p*r + 1.3981999507e-3f ;
    p = p*r + 8.3334519073e-3f ;
    p = p*r + 4.1665795894e-2f ;
    p = p*r + 1.6666665459e-1f ;
    p = p*r + 5.0000001201e-1f ;
    p = p*r*r + r + 1.0f ;
    int e = ((int)n + 127) << 23 ;
    float s ;
    memcpy(&s, &e, sizeof(float)) ;
    return p*s ;
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)BR * BC * sizeof(float));
    float *Kt = malloc((size_t)d * (N+16) * sizeof(float)) ;

    /* 1. S = Q K^T * scale */
    for(int j = 0 ; j < N ; j++){
        for(int k = 0 ; k < d ; k++){
            Kt[(size_t)k*(N+16) + j] = K[(size_t)j*d + k] ;
        }
    }

    for(int i0 = 0 ; i0 < N ; i0 += BR){
        float m[BR], l[BR] ;
        for(int i = 0 ; i < BR ; i++){
            m[i] = -INFINITY ;
            l[i] = 0.0f ;
        }
        memset(O + (size_t)i0*d, 0, (size_t)BR*d*sizeof(float)) ;

        for(int j0 = 0 ; j0 < N ; j0 += BC){

            for(int j = 0 ; j < BC ; j += C){
                for(int i = 0 ; i < BR ; i += R){
                    float acc[R][C] = {{0}} ;
                    for(int k = 0 ; k < d ; k++){
                        for(int r = 0 ; r < R ; r++){
                            float q = Q[(size_t)(i0+i+r)*d + k] ;
                            for(int c = 0 ; c < C ; c++){
                                acc[r][c] += q*Kt[(size_t)k*(N+16) + j0 + j + c] ;
                            }
                        }
                    }
                    for(int r = 0 ; r < R ; r++){
                        for(int c = 0 ; c < C ; c++){
                            S[(size_t)(i+r)*BC + j + c] = acc[r][c] * scale ;
                        }
                    }
                }
            }

            /* 2. softmax of each row of S */
            for(int i = 0 ; i < BR ; i++){
                float *row = S + (size_t)i*BC ;
                float mx[8] ;
                for(int c = 0 ; c < 8 ; c++){
                    mx[c] = row[c] ;
                }
                for(int j = 8 ; j < BC ; j += 8){
                    for(int c = 0 ; c < 8 ; c++){
                        mx[c] = row[j+c] > mx[c] ? row[j+c] : mx[c] ;
                    }
                }
                float mnew = m[i] ;
                for(int c = 0 ; c < 8 ; c++){
                    if(mx[c] > mnew) mnew = mx[c] ;
                }
                float sm[8] = {0} ;
                for(int j = 0 ; j < BC ; j += 8){
                    for(int c = 0 ; c < 8 ; c++){
                        float e = fast_exp(row[j+c] - mnew) ;
                        row[j+c] = e ;
                        sm[c] += e ;
                    }
                }
                float sum = 0.0f ;
                for(int c = 0 ; c < 8 ; c++){
                    sum += sm[c] ;
                }
                float alpha = fast_exp(m[i] - mnew) ;
                l[i] = alpha*l[i] + sum ;
                m[i] = mnew ;
                float *o = O + (size_t)(i0+i)*d ;
                for(int k = 0 ; k < d ; k++){
                    o[k] *= alpha ;
                }
            }

            /* 3. O = S V */

            for(int k0 = 0 ; k0 < d ; k0 += C){
                for(int i = 0 ; i < BR ; i += R){
                    float acc[R][C] = {{0}} ;
                    for(int j = 0 ; j < BC ; j++){
                        for(int r = 0 ; r < R ; r++){
                            float q = S[(size_t)(i+r)*BC + j] ;
                            for(int c = 0 ; c < C ; c++){
                                acc[r][c] += q*V[(size_t)(j0+j)*d + k0 + c] ;
                            }
                        }
                    }
                    for(int r = 0 ; r < R ; r++){
                        for(int c = 0 ; c < C ; c++){
                            O[(size_t)(i0+i+r)*d + k0 + c] += acc[r][c] ;
                        }
                    }
                }
            }
        }

        for(int i = 0 ; i < BR ; i++){
            float inv = 1.0f / l[i] ;
            for(int k = 0 ; k < d ; k++){
                O[(size_t)(i0+i)*d + k] *= inv ;
            }
        }
    }
    free(Kt) ;
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
