/// File: linear_regression_sample_main.c
/// Author: Dane Thompson
/// Date: 9/26/2024
/// Description: Sample for using LinearRegression library

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "LinearRegression.h"

#define OBSERVATIONS 100
#define FEATURES 5

// For estimation
double X[OBSERVATIONS][FEATURES];
double Y[OBSERVATIONS];
double betas[FEATURES];

// For performance checking
double residuals[OBSERVATIONS];

int main() {
    printf("Linear regression example:\n");

    // Set RNG seed
    srand(1234);

    // Generate observation matrix X and output Y
    for (uint16_t row = 0; row < OBSERVATIONS; row++) {
        // For polynomial regression: each column represents 1, t, t^2, t^3, ...
        for (uint8_t col = 0; col < FEATURES; col++) {
            X[row][col] = pow(row, col);
        }

        // Example output vector for testing
        Y[row] = -17+1.0*pow(row,1)+6.9*pow(row,2)-0.67*pow(row,3)+0.3*pow(row,4);
        Y[row] += 1.0*((float)rand()/(float)RAND_MAX - 0.5);  // Add some randomness so it's not perfect
        
    }

    // Fit linear regression model
    int err = Linear_Regression(OBSERVATIONS, FEATURES, X, betas, Y);
    if (err) {
        printf("Error in Linear_Regression: %d\n", err);
        return 0;
    }

    // Print estimated beta coefficients
    printf("Betas: ");
    for (uint8_t i = 0; i < FEATURES-1; i++) {
        printf("%f, ", betas[i]);
    }
    printf("%f\n", betas[FEATURES-1]);

    // Compute residuals and RMSE and R_Squared values
    Compute_Residuals(OBSERVATIONS, FEATURES, residuals, X, betas, Y);
    double rmse, r_squared;
    Compute_RMSE_and_RSquared(OBSERVATIONS, residuals, Y, &rmse, &r_squared);
    printf("RMSE: %f \nR_sqrd: %f", rmse, r_squared);

    return 0;
}
