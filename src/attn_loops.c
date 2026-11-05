/*
 * attn_loops.c -- Q3a: loop reordering so the compiler can vectorise.
 *
 * The two matmuls are reordered so that the innermost loop is not a reduction
 * and walks memory contiguously: step 1 uses i->k->j over a transposed copy
 * of K, step 3 uses i->j->k. Softmax is unchanged.
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
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kt = malloc((size_t)d * N * sizeof(float)); /* K transposed: d x N */

    /* 0. Kt[k][j] = K[j][k], so step 1 can read K contiguously along j.
     *    O(N d) work, negligible next to the O(N^2 d) matmuls. */
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    /* 1. S = Q K^T * scale, loop order i -> k -> j.
     *    Inner j loop: S[i][j] += Q[i][k] * Kt[k][j]. Each j writes a different
     *    S[i][j] (no reduction) and both S and Kt are contiguous in j, so it
     *    vectorises without reordering any sum. */
    for (int i = 0; i < N; i++) {
        float *Si = S + (size_t)i * N;
        for (int j = 0; j < N; j++)
            Si[j] = 0.0f;
        for (int k = 0; k < d; k++) {
            const float q = Q[(size_t)i * d + k];
            const float *Ktk = Kt + (size_t)k * N;
            for (int j = 0; j < N; j++)
                Si[j] += q * Ktk[j];
        }
        for (int j = 0; j < N; j++)
            Si[j] *= scale;
    }

    /* 2. softmax of each row of S (unchanged) */
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; j++)
            row[j] = row[j] / sum;
    }

    /* 3. O = S V, loop order i -> j -> k.
     *    Inner k loop: O[i][k] += S[i][j] * V[j][k]. Each k writes a different
     *    O[i][k] and both O and V are contiguous in k. */
    for (int i = 0; i < N; i++) {
        float *Oi = O + (size_t)i * d;
        for (int k = 0; k < d; k++)
            Oi[k] = 0.0f;
        for (int j = 0; j < N; j++) {
            const float s = S[(size_t)i * N + j];
            const float *Vj = V + (size_t)j * d;
            for (int k = 0; k < d; k++)
                Oi[k] += s * Vj[k];
        }
    }

    free(Kt);
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
