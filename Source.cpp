//#include<stdio.h>
//#include<iostream>
////#include<string>
////#include<fstream>
//#include<math.h>
//#include<time.h>
//
//using namespace std;
//
//float compute_error_for_line_given_points(double *b, double *m, float *pointX, float *pointY, int num_points)
//{
//	float totalError = 0;
//	float ret = 0;
//	for (int i = 0; i<num_points; i++)
//	{
//		float x = pointX[i];
//		float y = pointY[i];
//		totalError += powf((y - (*m * x + *b)), 2.0);
//	}
//	ret = (totalError / float(num_points));
//	return ret;
//}
//
//void step_gradient(double *new_b, double *new_m, double *b_current, double *m_current, float *pointX, float *pointY, float learningRate, int num_points) {
//	float b_gradient = 0.0;
//	float m_gradient = 0.0;
//	float N = float(num_points);
//
//	for (int i = 0; i < num_points; i++) {
//		float x = pointX[i];
//		float y = pointY[i];
//		b_gradient += -(2 / N) * (y - ((*m_current * x) + *b_current));
//		m_gradient += -(2 / N) * x * (y - ((*m_current * x) + *b_current));
//		*new_b = *b_current - (learningRate * b_gradient);
//		*new_m = *m_current - (learningRate * m_gradient);
//	}
//	//printf("debug");
//}
//
//void gradient_descent_runner(double *b, double *m, float *pointX, float *pointY, float starting_b, float starting_m, float learning_rate, int num_iterations)
//{
//	*b = starting_b;
//	*m = starting_m;
//	double new_b = 0;
//	double new_m = 0;
//
//	for (int i = 0; i < num_iterations; i++)
//	{
//		step_gradient(&new_b, &new_m, b, m, pointX, pointY, learning_rate, 100);
//		*b = new_b;
//		*m = new_m;
//	}
//}
//
//int main()
//{
//	//check for time interval
//	clock_t begin, end;
//	//start time
//	begin = clock();
//	
//	float f1, f2;
//	float pointX[100], pointY[100];
//	FILE *fp;
//	
//	// read CVS
//	fp = fopen("C:/Users/user/Desktop/data.csv", "r");
//	int i = 0;
//	while (fscanf(fp, "%g,%g\n", &f1, &f2) == 2)
//	{
//		pointX[i] = f1;
//		pointY[i] = f2;
//		//printf("%g, %g\n", f1, f2);
//		i++;
//	}
//
//	float learning_rate = 0.0001;
//	double initial_b = 0;
//	double initial_m = 0;
//	int num_iterations = 1000000;
//	float error = 0;
//	double b = 0.0;
//	double m = 0.0;
//	//calculate first total error
//	error = compute_error_for_line_given_points(&initial_b, &initial_m, pointX, pointY, 100);
//	printf("Starting gradient descent at b = %f, m = %f, error = %f\n", initial_b, initial_m, error);
//	printf("Running...\n");
//	//calculation and update weight and bias.
//	gradient_descent_runner(&b, &m, pointX, pointY, initial_b, initial_m, learning_rate, num_iterations);
//	//calculate error after backpropagation
//	error = compute_error_for_line_given_points(&b, &m, pointX, pointY, 100);
//	printf("After %d iterations b = %f, m = %f, error = %f\n", num_iterations, b, m, error);
//	
//	//end time
//	end = clock();
//	printf("GPU time inverval : %d msec\n", (end - begin));
//	return 0;
//}

#include<stdio.h>
#include<iostream>
//#include<string>
//#include<fstream>
#include<math.h>
#include<time.h>

using namespace std;

class NeuralNetwork {
public:
	float synaptic_weights;
	NeuralNetwork();
	void __sigmoid(float x);
	// The derivative of the Sigmoid function.
	// This is the gradient of the Sigmoid curve.
	// It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x) :
	return x * (1 - x)
	// We train the neural network through a process of trial and error.
	// Adjusting the synaptic weights each time.
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations) :
	for iteration in xrange(number_of_training_iterations) :
		// Pass the training set through our neural network(a single neuron).
		output = self.think(training_set_inputs)
		// Calculate the error(The difference between the desired output
		// and the predicted output).
		error = training_set_outputs - output
		// Multiply the error by the input and again by the gradient of the Sigmoid curve.
		// This means less confident weights are adjusted more.
		// This means inputs, which are zero, do not cause changes to the weights.
		adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
		// Adjust the weights.
		self.synaptic_weights += adjustment
		// The neural network thinks.
		def think(self, inputs) :
		// Pass inputs through our neural network(our single neuron).
			return self.__sigmoid(dot(inputs, self.synaptic_weights))
};
NeuralNetwork::NeuralNetwork() {
	// Seed the random number generator, so it generates the same numbers
	// every time the program runs.
	srand(time(NULL));
	// We model a single neuron, with 3 input connections and 1 output connection.
	// We assign random weights to a 3 x 1 matrix, with values in the range - 1 to 1
	// and mean 0.
	
	this->synaptic_weights = 2 * random.random((3, 1)) - 1;
};
// The Sigmoid function, which describes an S shaped curve.
// We pass the weighted sum of the inputs through this function to
// normalise them between 0 and 1.
def __sigmoid(self, x) :
	return 1 / (1 + exp(-x))
int main() {
	//Intialise a single neuron neural network.
	neural_network = NeuralNetwork()
	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights
	// The training set.We have 4 examples, each consisting of 3 input values
	// and 1 output value.
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T
	// Train the neural network using a training set.
	// Do it 10, 000 times and make small adjustments each time.
	neural_network.train(training_set_inputs, training_set_outputs, 10000)
	print "New synaptic weights after training: "
	print neural_network.synaptic_weights
	// Test the neural network with a new situation.
	print "Considering new situation [1, 0, 0] -> ?: "
	print neural_network.think(array([1, 0, 0]))
	return 0;
}