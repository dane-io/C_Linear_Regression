# C_Linear_Regression
This code is a C implementation for linear regression using least-squares using LU decomposition. I wrote this to use for embedded systems (STM32), but it should be fine for any C compiler that supports VLA in the function paramaters. The main limitations for this code is the number of observations and features. Currently, the number of observations is stored as unsigned 16 bit and features as unsigned 8 bit. If you wanted more features, you can make these larger data types (note, there's a sneaky int16_t somewhere in the code that you'll need to change also).

I've added a linear_regression_sample_main.c file to show how to implement the functions used here.

As always, use at your own risk.

Dane Thompson
