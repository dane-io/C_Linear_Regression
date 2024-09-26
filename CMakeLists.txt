cmake_minimum_required(VERSION 3.20.0)

project(Regression_Sample)

set(SOURCES
    linear_regression_sample_main.c
    LinearRegression.c
)

add_executable(Regression_Sample ${SOURCES})