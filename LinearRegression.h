/// File: LinearRegression.h
/// Author: Dane Thompson
/// Date: 9/26/2024
/// Description: Header for C library to perform linear regression for arbitrary number of observations/features. (Designed for embedded)
#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <stdint.h>

/// @brief Decompose matrix A into LU using partial pivoting
/// @param size number of rows or columns of A, P, L, and U (assumed square)
/// @param P pivot matrix, [row][col] format with dimensions size x size
/// @param A in [row][col] format with dimensions size x size
/// @param L lower matrix, [row][col] format with dimensions size x size
/// @param U upper matrix, [row][col] format with dimensions size x size
/// @return 0 for success, -1 for singular matrix
int LU_Decomposition(uint8_t size, uint8_t P[size][size], double A[size][size], double L[size][size], double U[size][size]);

/// @brief Solve for d in Ld=Pb using forward substition
/// @param size number of rows or columns of A, P, L, and U (assumed square)
/// @param L lower matrix, [row][col] format with dimensions size x size
/// @param d temporary vector (for Ld=Pb) with dimensions size x 1
/// @param P pivot matrix, [row][col] format with dimensions size x size
/// @param b column vector with dimensions size x 1
/// @return 0 for success
int Forward_Substition(uint8_t size, double L[size][size], double* d, uint8_t P[size][size], double* b);

/// @brief solve for x in Ux=d using backward substition
/// @param size number of rows or columns of A, P, L, and U (assumed square)
/// @param U upper matrix, [row][col] format with dimensions size x size
/// @param x vector for Ux=d which also solves Ax=b with dimensions size x 1
/// @param d temporary vector (for Ld=Pb) with dimensions size x 1
/// @return 0 for success, -1 for singular U matrix
int Backward_Substition(uint8_t size, double U[size][size], double* x, double* d);

/// @brief fit linear regression model using least-squares and LU decomposition
/// @param X_rows number of rows (observations) in observation matrix
/// @param X_cols number of columns (features) in observation matrix
/// @param X observation matrix[row][col] with dimensions X_rows x X_cols
/// @param betas vector for weights of features (the x in LU Decomp) to be outputted in with dimensions X_cols x 1
/// @param Y vector of output values for each observation with dimensions X_rows x 1
/// @return 0 for success, -1 for error in LU decomposition, -2 for error in forward substition, -3 for error in backward substition
int Linear_Regression(uint16_t X_rows, uint8_t X_cols, double X[X_rows][X_cols], double* betas, double* Y);

/// @brief compute residual errors for every observation
/// @param X_rows number of rows (observations) in observation matrix
/// @param X_cols number of columns (features) in observation matrix
/// @param residuals vector for difference between estimated Ys (Y_hat) using estimated betas with dimensions X_rows x 1
/// @param X observation matrix[row][col] with dimensions X_rows x X_cols
/// @param betas vector for weights of features (the x in LU Decomp) to be outputted in with dimensions X_cols x 1
/// @param Y vector of output values for each observation with dimensions X_rows x 1
/// @return 0 for success
int Compute_Residuals(uint16_t X_rows, uint8_t X_cols, double* residuals, double X[X_rows][X_cols], double* betas, double* Y);

/// @brief compute root mean squared error (RMSE) and R Squared value for linear regression model
/// @param length number of observations (and number of residuals)
/// @param residuals vector for difference between estimated Ys (Y_hat) using estimated betas with dimensions X_rows x 1
/// @param Y vector of output values for each observation with dimensions X_rows x 1
/// @param rmse pointer to save root mean squared error (RMSE)
/// @param r_squared pointer to save R Squared value
/// @return 0 for success
int Compute_RMSE_and_RSquared(uint16_t length, double* residuals, double* Y, double* rmse, double* r_squared);

#endif
