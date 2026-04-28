#include <iostream>
#include <vector>
#include <string>
#include <emmintrin.h>
#include "The-Tips.h"

using namespace std;

double TheTips::solve(vector<string> clues, vector<int> probability, int print)
{
    int n = clues.size();

    // Calculate row size in bytes, rounded up to a multiple of 16 for SIMD alignment/safety
    int row_sz_bytes = ((n + 127) / 128) * 16;
    vector<uint8_t> packed_matrix(n * row_sz_bytes, 0);

    // Pack strings into bits
    for (int i = 0; i < n; i++)
    {
        // Every egg can reach itself
        packed_matrix[i * row_sz_bytes + (i / 8)] |= (1 << (i % 8));
        for (int j = 0; j < n; j++)
        {
            if (clues[i][j] == 'Y')
            {
                packed_matrix[i * row_sz_bytes + (j / 8)] |= (1 << (j % 8));
            }
        }
    }

    // Floyd-Warshall with SIMD
    for (int v = 0; v < n; v++)
    {
        uint8_t *row_v = &packed_matrix[v * row_sz_bytes];
        for (int i = 0; i < n; i++)
        {
            uint8_t *row_i = &packed_matrix[i * row_sz_bytes];
            // Check if egg i can reach egg v
            if (row_i[v / 8] & (1 << (v % 8)))
            {
                // OR row i with row v using 128-bit chunks
                for (int j = 0; j < row_sz_bytes; j += 16)
                {
                    __m128i vi = _mm_loadu_si128((__m128i *)(row_i + j));
                    __m128i vv = _mm_loadu_si128((__m128i *)(row_v + j));
                    _mm_storeu_si128((__m128i *)(row_i + j), _mm_or_si128(vi, vv));
                }
            }
        }
    }

    // Calculate probabilities
    vector<double> p(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        double prob_i = probability[i] / 100.0;
        for (int j = 0; j < n; j++)
        {
            // If i can reach j
            if (packed_matrix[i * row_sz_bytes + (j / 8)] & (1 << (j % 8)))
            {
                p[j] += (1.0 - p[j]) * prob_i;
            }
        }
    }

    double expected_value = 0;
    for (int i = 0; i < n; i++)
        expected_value += p[i];
    return expected_value;
}