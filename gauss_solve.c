/*----------------------------------------------------------------
* File:     gauss_solve.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:29 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
#include "gauss_solve.h"
#include <math.h>

void gauss_solve_in_place(const int n, double A[n][n], double b[n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k+1; i < n; ++i) {
      /* Store the multiplier into A[i][k] as it would become 0 and be
	 useless */
      A[i][k] /= A[k][k];
      for( int j = k+1; j < n; ++j) {
	A[i][j] -= A[i][k] * A[k][j];
      }
      b[i] -= A[i][k] * b[k];
    }
  } /* End of Gaussian elimination, start back-substitution. */
  for(int i = n-1; i >= 0; --i) {
    for(int j = i+1; j<n; ++j) {
      b[i] -= A[i][j] * b[j];
    }
    b[i] /= A[i][i];
  } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[n][n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	/* U[k][i] -= L[k][j] * U[j][i] */
	A[k][i] -=  A[k][j] * A[j][i]; 
      }
    }
    for(int i = k+1; i<n; ++i) {
      for(int j=0; j<k; ++j) {
	/* L[i][k] -= A[i][k] * U[j][k] */
	A[i][k] -= A[i][j]*A[j][k]; 
      }
      /* L[k][k] /= U[k][k] */
      A[i][k] /= A[k][k];	
    }
  }
}
//BLOCK

// Function to swap rows in a matrix
void swapRows(double* matrix, int n, int row1, int row2) {
    for (int j = 0; j < n; j++) {
        double temp = matrix[row1 * n + j];
        matrix[row1 * n + j] = matrix[row2 * n + j];
        matrix[row2 * n + j] = temp;
    }
}

void plu(int n, double A[n][n], int P[n]) {
    // Create a temporary array for row swapping
    double temp_row[n];

    // Initialize the permutation vector P
    for (int i = 0; i < n; i++) {
        P[i] = i;  // Identity matrix for permutation
    }

    // Initialize L as a zero matrix and U as a copy of A
    double L[n][n];  // Lower triangular matrix
    double U[n][n];  // Upper triangular matrix

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = A[i][j];
        }
    }

    // Find pivot (max absolute value in column k from row k to n)
        int pivotRow = k;
        for (int i = k + 1; i < n; i++) {
            if (fabs(U[i * n + k]) > fabs(U[pivotRow * n + k])) {
                pivotRow = i;
            }
        }

        // If pivotRow != k, swap rows k and pivotRow in U, P, and L
        if (pivotRow != k) {
            swapRows(U, n, k, pivotRow);
            swapRows(P, n, k, pivotRow);

            // In L, we swap only the elements up to column k-1
            for (int i = 0; i < k; i++) {
                double temp = L[k * n + i];
                L[k * n + i] = L[pivotRow * n + i];
                L[pivotRow * n + i] = temp;
            }
        }

        // Perform the elimination process
        for (int i = k + 1; i < n; i++) {
            double multiplier = U[i][k] / U[k][k];
            L[i][k] = multiplier;

            // Update the U matrix by eliminating the entries below the pivot
            for (int j = k; j < n; j++) {
                U[i][j] -= multiplier * U[k][j];
            }
        }
    }

    // Set the diagonal of L to 1
    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;
    }
}
    
//BLOCK

void lu_in_place_reconstruct(int n, double A[n][n])
{
  for(int k = n-1; k >= 0; --k) {
    for(int i = k+1; i<n; ++i) {
      A[i][k] *= A[k][k];
      for(int j=0; j<k; ++j) {
	A[i][k] += A[i][j]*A[j][k];
      }
    }
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	A[k][i] +=  A[k][j] * A[j][i];
      }
    }
  }
}
