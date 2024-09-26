/// File: LinearRegression.c
/// Author: Dane Thompson
/// Date: 9/26/2024
/// Description: C library to perform linear regression for arbitrary number of observations/features. (Designed for embedded)

#include <math.h>
#include <stdint.h>

/// @brief Decompose matrix A into LU using partial pivoting
/// @param size number of rows or columns of A, P, L, and U (assumed square)
/// @param P pivot matrix, [row][col] format with dimensions size x size
/// @param A in [row][col] format with dimensions size x size
/// @param L lower matrix, [row][col] format with dimensions size x size
/// @param U upper matrix, [row][col] format with dimensions size x size
/// @return 0 for success, -1 for singular matrix
int LU_Decomposition(uint8_t size, uint8_t P[size][size], double A[size][size], double L[size][size], double U[size][size]) {
    // Make U equal to A at beginning and do operations on later (this way user can keep original copy of A)
    // Make L equal to all zeros at beginning
    // Make P equal to all zeros at beginning (add identity matrix after)
    for (uint8_t row = 0; row < size; row++) {
        for (uint8_t col = 0; col < size; col++) {
            U[row][col] = A[row][col];
            L[row][col] = 0;
            P[row][col] = 0;
        }
    }

    // Assign identity matrix to P
    for (uint8_t i = 0; i < size; i++) {
        P[i][i] = 1;
    }
    
    // Loop through each column step
    for (uint8_t col_step = 0; col_step < size; col_step++) {
        // Calculate max absolute value for given column of U (which is initially a copy of A)
        double max = 0;
        uint8_t max_row = 0;
        for (uint8_t row = col_step; row < size; row++) {
            if (fabs(U[row][col_step]) >= max) {
                max = fabs(U[row][col_step]);
                max_row = row;
            }
        }

        // After max has been computed, swap row with max in it and row = col_step in P, L, and U
        uint8_t old_P_row[size];  // Row containing copy of row = col_step
        double old_L_row[size];  // Row containing copy of row = col_step
        double old_U_row[size];  // Row containing copy of row = col_step
        for (uint8_t i = 0; i < size; i++) {
            // Make copy of row in P
            old_P_row[i] = P[col_step][i];
            // Swap rows in P to record pivots about to be made
            P[col_step][i] = P[max_row][i];
            P[max_row][i] = old_P_row[i];

            // Make copy of row in L
            old_L_row[i] = L[col_step][i];
            // Swap rows in L
            L[col_step][i] = L[max_row][i];
            L[max_row][i] = old_L_row[i];

            // Make copy of row in U
            old_U_row[i] = U[col_step][i];
            // Swap rows in L
            U[col_step][i] = U[max_row][i];
            U[max_row][i] = old_U_row[i];
        }

        // Calculate rows of L for given column step and cancel rows of A to make U (do the operation on U since it's a copy of A)
        for (uint8_t row = col_step+1; row < size; row++) {   // When col_step+1 == size, nothing will happen since condition false
            // Check for zeros on diagonal (this implies singular matrix) and return so divide by 0 doesn't occur
            if (U[col_step][col_step] == 0) {
                return -1;
            }
            
            // Set values in L
            L[row][col_step] = U[row][col_step] / U[col_step][col_step];

            // Cancel rows in U (A)
            for (uint8_t col = col_step; col < size; col++) {
                U[row][col] = U[row][col] - L[row][col_step]*U[col_step][col];
            }
        }

    }   // End of looping through each column step

    // Set diagonal of L to be all 1s
    for (uint8_t i = 0; i < size; i++) {
        L[i][i] = 1;
    }

    // Return 0 for success
    return 0;
}   // End of LU_Decomposition

/// @brief Solve for d in Ld=Pb using forward substition
/// @param size number of rows or columns of A, P, L, and U (assumed square)
/// @param L lower matrix, [row][col] format with dimensions size x size
/// @param d temporary vector (for Ld=Pb) with dimensions size x 1
/// @param P pivot matrix, [row][col] format with dimensions size x size
/// @param b column vector with dimensions size x 1
/// @return 0 for success
int Forward_Substition(uint8_t size, double L[size][size], double* d, uint8_t P[size][size], double* b) {
    // Initialize d to zeros
    for (uint8_t i = 0; i < size; i++) {
        d[i] = 0;
    }

    // First do matrix multiplication of P*b
    double Pb[size];
    for (uint8_t row = 0; row < size; row++) {
        Pb[row] = 0;
        for (uint8_t col = 0; col < size; col++) {
            Pb[row] += P[row][col]*b[col];
        }
    }

    // Calculate rows of d using Forward Substition
    for (uint8_t row = 0; row < size; row++) {
        double Ld_dot = 0;
        // Compute L[row][:]*d[:] (need to do it this way so d[row] isn't changed during dot product)
        for (uint8_t col = 0; col < size; col++) {
            Ld_dot += L[row][col]*d[col];
        }
        d[row] = Pb[row] - Ld_dot;
    }

    // Return 0 for success
    return 0;
}   // End of Forward_Substition


/// @brief solve for x in Ux=d using backward substition
/// @param size number of rows or columns of A, P, L, and U (assumed square)
/// @param U upper matrix, [row][col] format with dimensions size x size
/// @param x vector for Ux=d which also solves Ax=b with dimensions size x 1
/// @param d temporary vector (for Ld=Pb) with dimensions size x 1
/// @return 0 for success, -1 for singular U matrix
int Backward_Substition(uint8_t size, double U[size][size], double* x, double* d) {
    // Initialize x to zeros
    for (uint8_t i = 0; i < size; i++) {
        x[i] = 0;
    }

    // Check for zeros on diagonal of U which could cause divide by 0 undefined behavior
    for (uint8_t i = 0; i < size; i++) {
        if (U[i][i] == 0) {
            return -1;
        }
    }

    // Since this works backwards, go ahead and assign last value of x (since last diagonal of U is only element to consider)
    x[size-1] = d[size-1] / U[size-1][size-1];

    // Loop through U and d backwards (enxcluding last element) and assign x
    for (int16_t row = size-2; row >= 0; row--) {   // Using int16_t here since size is actually uint8_t and row should only go to -1. If using uint, wrap around would occur.
        double Ux_dot = 0;
        for (uint8_t col = row+1; col < size; col++) {
            Ux_dot += U[row][col]*x[col];   // Doing a dot product here
        }
        x[row] = (d[row] - Ux_dot) / U[row][row];
    }

    // Return 0 for success
    return 0;
}   // End of Backward_Substition


/// @brief fit linear regression model using least-squares and LU decomposition
/// @param X_rows number of rows (observations) in observation matrix
/// @param X_cols number of columns (features) in observation matrix
/// @param X observation matrix[row][col] with dimensions X_rows x X_cols
/// @param betas vector for weights of features (the x in LU Decomp) to be outputted in with dimensions X_cols x 1
/// @param Y vector of output values for each observation with dimensions X_rows x 1
/// @return 0 for success, -1 for error in LU decomposition, -2 for error in forward substition, -3 for error in backward substition
int Linear_Regression(uint16_t X_rows, uint8_t X_cols, double X[X_rows][X_cols], double* betas, double* Y) {
    // Create X'X matrix (becomes the A in LU Decomp)
    uint8_t size = X_cols;  // Size for the square matrices used later
    double X_T_X[size][size];
    for (uint8_t row = 0; row < size; row++) {
        for (uint8_t col = 0; col < size; col++) {
            X_T_X[row][col] = 0;
            for (uint16_t i = 0; i < X_rows; i++) {
                X_T_X[row][col] += X[i][row]*X[i][col];
            }
        }
    }

    // Create X'Y matrix (becomes the b in LU Decomp)
    double X_T_Y[size];
    for (uint8_t row = 0; row < size; row++) {
        X_T_Y[row] = 0;
        for (uint16_t i = 0; i < X_rows; i++) {
            X_T_Y[row] += X[i][row]*Y[i];
        }
    }

    // Define matrices/vectors to be used
    uint8_t P[size][size];
    double L[size][size];
    double U[size][size];
    double d[size];

    if (LU_Decomposition(size, P, X_T_X, L, U) != 0) {
        return -1;
    }
    if (Forward_Substition(size, L, d, P, X_T_Y) != 0) {
        return -2;
    }
    if (Backward_Substition(size, U, betas, d) != 0) {
        return -3;
    }
    // Return 0 for success
    return 0;
}   // End of Linear_Regression


/// @brief compute residual errors for every observation
/// @param X_rows number of rows (observations) in observation matrix
/// @param X_cols number of columns (features) in observation matrix
/// @param residuals vector for difference between estimated Ys (Y_hat) using estimated betas with dimensions X_rows x 1
/// @param X observation matrix[row][col] with dimensions X_rows x X_cols
/// @param betas vector for weights of features (the x in LU Decomp) to be outputted in with dimensions X_cols x 1
/// @param Y vector of output values for each observation with dimensions X_rows x 1
/// @return 0 for success
int Compute_Residuals(uint16_t X_rows, uint8_t X_cols, double* residuals, double X[X_rows][X_cols], double* betas, double* Y) {
    // Create y_hat to store estimate Y values
    double Y_hat[X_rows];

    // Calculate Y_hat and residuals
    for (uint16_t row = 0; row < X_rows; row++) {
        Y_hat[row] = 0;
        for (uint8_t col = 0; col < X_cols; col++) {
            Y_hat[row] += X[row][col]*betas[col];
        }
        residuals[row] = Y_hat[row] - Y[row];
    }

    // Return 0 for success
    return 0;
}   // End of Compute_Residuals


/// @brief compute root mean squared error (RMSE) and R Squared value for linear regression model
/// @param length number of observations (and number of residuals)
/// @param residuals vector for difference between estimated Ys (Y_hat) using estimated betas with dimensions X_rows x 1
/// @param Y vector of output values for each observation with dimensions X_rows x 1
/// @param rmse pointer to save root mean squared error (RMSE)
/// @param r_squared pointer to save R Squared value
/// @return 0 for success
int Compute_RMSE_and_RSquared(uint16_t length, double* residuals, double* Y, double* rmse, double* r_squared) {
    // Calculate RMSE
    for (uint16_t i = 0; i < length; i++) {
        *rmse += residuals[i]*residuals[i]; // Square residuals
    }
    double sum_sqr_residuals = *rmse;       // Used for RSquared calculation later
    *rmse /= (double)length;                // Compute mean
    *rmse = sqrt(*rmse);                    // Compute root

    // Calculate Y_avg (used for RSquared)
    double Y_avg = 0;
    for (uint16_t i = 0; i < length; i++) {
        Y_avg += Y[i];
    }
    Y_avg /= (double)length;

    // Calculate total sum of squares
    double tot_sum_squares = 0;
    for (uint16_t i = 0; i < length; i++) {
        tot_sum_squares += (Y[i]-Y_avg)*(Y[i]-Y_avg);
    }

    // Calculate RSquared
    *r_squared = 1.0 - sum_sqr_residuals / tot_sum_squares;

    // Return 0 for success
    return 0;
}   // End of Compute_RMSE_and_RSquared
