#include<stdio.h>
#include<iostream>
//#include<string>
//#include<fstream>
#include<math.h>
#include<time.h>

using namespace std;

float compute_error_for_line_given_points(double *b, double *m, float *pointX, float *pointY, int num_points)
{
	float totalError = 0;
	float ret = 0;
	for (int i = 0; i<num_points; i++)
	{
		float x = pointX[i];
		float y = pointY[i];
		totalError += powf((y - (*m * x + *b)), 2.0);
	}
	ret = (totalError / float(num_points));
	return ret;
}

void step_gradient(double *new_b, double *new_m, double *b_current, double *m_current, float *pointX, float *pointY, float learningRate, int num_points) {
	float b_gradient = 0.0;
	float m_gradient = 0.0;
	float N = float(num_points);

	for (int i = 0; i < num_points; i++) {
		float x = pointX[i];
		float y = pointY[i];
		b_gradient += -(2 / N) * (y - ((*m_current * x) + *b_current));
		m_gradient += -(2 / N) * x * (y - ((*m_current * x) + *b_current));
		*new_b = *b_current - (learningRate * b_gradient);
		*new_m = *m_current - (learningRate * m_gradient);
	}
	//printf("debug");
}

void gradient_descent_runner(double *b, double *m, float *pointX, float *pointY, float starting_b, float starting_m, float learning_rate, int num_iterations)
{
	*b = starting_b;
	*m = starting_m;
	double new_b = 0;
	double new_m = 0;

	for (int i = 0; i < num_iterations; i++)
	{
		step_gradient(&new_b, &new_m, b, m, pointX, pointY, learning_rate, 100);
		*b = new_b;
		*m = new_m;
	}
}

int main()
{
	//check for time interval
	clock_t begin, end;
	//start time
	begin = clock();
	
	float f1, f2;
	float pointX[100], pointY[100];
	FILE *fp;
	
	// read CVS
	fp = fopen("C:/Users/user/Desktop/data.csv", "r");
	int i = 0;
	while (fscanf(fp, "%g,%g\n", &f1, &f2) == 2)
	{
		pointX[i] = f1;
		pointY[i] = f2;
		//printf("%g, %g\n", f1, f2);
		i++;
	}

	float learning_rate = 0.0001;
	double initial_b = 0;
	double initial_m = 0;
	int num_iterations = 1000000;
	float error = 0;
	double b = 0.0;
	double m = 0.0;
	//calculate first total error
	error = compute_error_for_line_given_points(&initial_b, &initial_m, pointX, pointY, 100);
	printf("Starting gradient descent at b = %f, m = %f, error = %f\n", initial_b, initial_m, error);
	printf("Running...\n");
	//calculation and update weight and bias.
	gradient_descent_runner(&b, &m, pointX, pointY, initial_b, initial_m, learning_rate, num_iterations);
	//calculate error after backpropagation
	error = compute_error_for_line_given_points(&b, &m, pointX, pointY, 100);
	printf("After %d iterations b = %f, m = %f, error = %f\n", num_iterations, b, m, error);
	
	//end time
	end = clock();
	printf("GPU time inverval : %d msec\n", (end - begin));
	return 0;
}