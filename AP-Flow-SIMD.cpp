#include <iostream>
#include <vector>
#include <emmintrin.h>
#include "AP-Flow.h"

void APFlow::CalcFlow()
{
    int total_elements = N * N;

    // Copy Adj to Flow
    for (int i = 0; i < total_elements; i++)
    {
        Flow[i] = Adj[i];
    }

    for (int v = 0; v < N; v++)
    {
        for (int i = 0; i < N; i++)
        {
            // Broadcast Flow[i][v] to all 16 lanes
            __m128i alli = _mm_set1_epi8(Flow[i * N + v]);

            uint8_t *row_i = &Flow[i * N];
            uint8_t *row_v = &Flow[v * N];

            for (int j = 0; j < N; j += 16)
            {
                // Load 16 bytes of Flow[v][j...j+15]
                __m128i vv = _mm_loadu_si128((__m128i *)(row_v + j));

                // Load 16 bytes of Flow[i][j...j+15]
                __m128i iv = _mm_loadu_si128((__m128i *)(row_i + j));

                // fv = min(Flow[i][v], Flow[v][j])
                __m128i fv = _mm_min_epu8(alli, vv);

                // rv = max(Flow[i][j], fv)
                __m128i rv = _mm_max_epu8(fv, iv);

                // Store result back
                _mm_storeu_si128((__m128i *)(row_i + j), rv);
            }
        }
    }
}