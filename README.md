# linear regression C++ with CUDA

# Introduction

this code is working visual studio for C++ code.
the 'Source.cpp' file is the linear regression for CPU.
the 'kernel.cu' file is the linear regression for GPU using CUDA.

## Overview
- compute_error_for_line_given_points : this is for calculation of distance between line and data point.
- step_gradient : this is for calculation of bias and weight gradient and return new bias and new weight.
- gradient_descent_runner : this is for update new bias and new weight.(backpropagation.)

## Dependencies

* stdio.h : standard lib.
* iostream : input output lib.
* math : mathmatic function lib.
* time : time lib.(for calculation of time interval)

## Usage

- make the new project in visual studio with CUDA lib.
- Add the file into your project and compile, build and run it.