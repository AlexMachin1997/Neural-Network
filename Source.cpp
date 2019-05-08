// ANN with added loops
/*
Amongst Many...

https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

https://www.nnwj.de/backpropagation.html
*/

#include "math.h"
#include <iostream>
#include <iomanip>
using namespace std;

void displayStuff(bool, int);
void feedForward();
void backProp();

float testPattern[16][4]{
	{ 0.05f, 0.05f, 0.05f, 0.05f },
{ 0.05f, 0.05f, 0.05f, 0.99f },
{ 0.05f, 0.05f, 0.99f, 0.05f },
{ 0.05f, 0.05f, 0.99f, 0.99f },
{ 0.05f, 0.99f, 0.05f, 0.05f },
{ 0.05f, 0.99f, 0.05f, 0.99f },
{ 0.05f, 0.99f, 0.99f, 0.05f },
{ 0.05f, 0.99f, 0.99f, 0.99f },
{ 0.99f, 0.05f, 0.05f, 0.05f },
{ 0.99f, 0.05f, 0.05f, 0.99f },
{ 0.99f, 0.05f, 0.99f, 0.05f },
{ 0.99f, 0.05f, 0.99f, 0.99f },
{ 0.99f, 0.99f, 0.05f, 0.05f },
{ 0.99f, 0.99f, 0.05f, 0.99f },
{ 0.99f, 0.99f, 0.99f, 0.05f },
{ 0.99f, 0.99f, 0.99f, 0.99f }
};

float testTarget[16][4]{
{ 0.05f,	0.05f, 0.05f, 0.99f },
{ 0.05f,	0.05f, 0.99f, 0.05f },
{ 0.05f,	0.05f, 0.99f, 0.99f },
{ 0.05f,	0.99f, 0.05f, 0.05f },
{ 0.05f,	0.99f, 0.05f, 0.99f },
{ 0.05f,	0.99f, 0.99f, 0.05f },
{ 0.05f,	0.99f, 0.99f, 0.99f },
{ 0.99f,	0.05f, 0.05f, 0.05f },
{ 0.99f,	0.05f, 0.05f, 0.99f },
{ 0.99f,	0.05f, 0.99f, 0.05f },
{ 0.99f,	0.05f, 0.99f, 0.99f },
{ 0.99f,	0.99f, 0.05f, 0.05f },
{ 0.99f,	0.99f, 0.05f, 0.99f },
{ 0.99f,	0.99f, 0.99f, 0.05f },
{ 0.99f,	0.99f, 0.99f, 0.99f },
{ 0.05f,	0.05f, 0.05f, 0.05f }
};

int numOfTrainingSets = 15; // Must be 1 less than the actual avaliable sets (Bias I think not sure :/)
const int numOfInputNodes = 4;
const int numOfHiddenNodes = 6;
const int numOfOutputNodes = 4;


float inputLayer[numOfInputNodes];
float hiddenLayer[numOfHiddenNodes];
float outputLayer[numOfOutputNodes];

float b1 = 0.35f;
float b2 = 0.6f;

float hiddenWeights[numOfInputNodes*numOfHiddenNodes];
float outputWeights[numOfHiddenNodes*numOfOutputNodes];

float totalError = 1;
float outputError[numOfOutputNodes*numOfHiddenNodes];
float hiddenError[numOfHiddenNodes*numOfInputNodes];

float target[numOfOutputNodes];
int queryResult[numOfOutputNodes];

float LC = 0.5f; //learning constant (Mutation value)

int epochCount = 1000000; // Max number of iterations 
int totalEpoch = epochCount; // used for the coutdown
int epochDisplayLimit = 100; // How many to display on screen. Limit the ouput to the screen as it will slow it down

int main() 
{

		// Random hidden weight
		for (int i = 0; i < numOfInputNodes*numOfHiddenNodes; i++) 
		{
			//hiddenWeights[i] = ((double)rand() / (RAND_MAX)) - 0.5f;
			hiddenWeights[i] = ((double)rand() / (RAND_MAX));
		}

		// Random output weight
		for (int i = 0; i < numOfHiddenNodes*numOfOutputNodes; i++) 
		{
			//outputWeights[i] = ((double)rand() / (RAND_MAX)) - 0.5f;
			outputWeights[i] = ((double)rand() / (RAND_MAX));
		}

		int failCount = 300;

		while (epochCount > 0 && failCount > 0)
		//while (epochCount > 0 && failCount > 0)
		{
			failCount--;
			
			//read in test data and target
			int rnd = (rand() % numOfTrainingSets);
			
			// rnd = 4; //TEMP!! REMOVE
			for (int i = 0; i < numOfInputNodes; i++) {
				inputLayer[i] = testPattern[rnd][i];
			}

			for (int i = 0; i < numOfInputNodes; i++) {
				target[i] = testTarget[rnd][i];
			}

			// Feed forward
			feedForward();

			// Back prop
			backProp();

			for (int i = 0; i < numOfInputNodes; i++) {
				if (outputLayer[i] >= 0.5f) {
					queryResult[i] = 1; 
				}
				else { 
					queryResult[i] = 0; 
				}
			}

			int sTag[numOfOutputNodes]; //sigmoid target grade

			// Sigmoid activiatin
			for (int i = 0; i < numOfOutputNodes; i++){
				if (target[i] >= 0.5f) { sTag[i] = 1; }
				else { sTag[i] = 0; }
			}

			bool achieved = true;
			for (int i = 0; i < numOfOutputNodes; i++) {
				if (queryResult[i] != sTag[i]) { achieved = false; failCount = 300; }
			}

			std::cout << std::fixed;
			std::cout << std::setprecision(2); // round output to 4 d.p.

			//output
			if (epochCount % epochDisplayLimit == 0) {
				displayStuff(achieved, failCount);
			}
			epochCount--;
		}

		cout << "\nEpoch Count: " << (totalEpoch - epochCount);

		cout << "\n\nFinal Test : \n";

		for (int tp = 0; tp < numOfTrainingSets; tp++) 
		{

			for (int i = 0; i < numOfInputNodes; i++) 
			{
				inputLayer[i] = testPattern[tp][i];
			}

			for (int i = 0; i < numOfOutputNodes; i++)
			{
				target[i] = testTarget[tp][i];
			}

			feedForward();
			for (int i = 0; i < numOfOutputNodes; i++) {
				if (outputLayer[i] >= 0.5f) { queryResult[i] = 1; }
				else { queryResult[i] = 0; }
			}

			int sTag[numOfOutputNodes];

			for (int i = 0; i < numOfOutputNodes; i++) {
				if (target[i] >= 0.5f) { sTag[i] = 1; }
				else { sTag[i] = 0; }
			}

			bool achieved = true; 

			for (int i = 0; i < numOfOutputNodes; i++) 
			{
				if (queryResult[i] != sTag[i]) { achieved = false; failCount = 100;}
			}

			std::cout << std::fixed;
			std::cout << std::setprecision(2); //round output 4 d.p

			displayStuff(achieved, 0);

		}

		//Delay
		char a;
		cin >> a;
	}


void displayStuff(bool achieved, int failcount) {

	cout << epochCount << ": ";

	cout << " I: ";
	for (int i = 0; i < numOfInputNodes; i++) {
		cout << inputLayer[i] << ",";
	}

	cout << " H: ";
	for (int i = 0; i < numOfHiddenNodes; i++) {
		cout << hiddenLayer[i] << ",";
	}

	cout << " O: ";
	for (int i = 0; i < numOfOutputNodes; i++) {
		cout << outputLayer[i] << ",";
	}

	cout << " T: ";
	for (int i = 0; i < numOfOutputNodes; i++) {
		cout << target[i] << ",";
	}

	cout << " Q: ";
	for(int i = 0; i < numOfOutputNodes; i++) {
		cout << queryResult[i] << ",";
	}

	std::cout << std::setprecision(6);
	cout << "TE: " << totalError;
	cout << " Achieved: " << achieved;
	cout << " FailCount: " << failcount;
	cout << endl;
}

void feedForward() {
	//FEED FORWARD
	//Input to hidden
	for (int i = 0; i < numOfHiddenNodes; i++) {
		float sum = 0;
		for (int j = 0; j < numOfInputNodes; j++)
		{
			sum += hiddenWeights[i*numOfInputNodes + j] * inputLayer[j];
		}
		sum += b1;
		//apply sigmoid and store 
		hiddenLayer[i] = 1 / (1 + exp(-sum));
	}

	//HIDEEN
	//hidden to output
	for (int i = 0; i < numOfOutputNodes; i++) {
		float sum = 0;
		for (int j = 0; j < numOfHiddenNodes; j++)
		{
			sum += outputWeights[i*numOfHiddenNodes + j] * hiddenLayer[j];
		}
		sum += b2;
		//apply sigmoid and store
		outputLayer[i] = 1 / (1 + exp(-sum));
	}
}

void backProp() {
	//calc total error - squared error
	totalError = 0;
	for (int i = 0; i < numOfOutputNodes; i++) {
		totalError += 0.5 * ((target[i] - outputLayer[i]) * (target[i] - outputLayer[i]));
	}

	int t2 = 0; // Temp variable
	// calc output error adjustment value for each weight
	for (int i = 0; i < numOfOutputNodes; i++)
	{
		for (int j = 0; j < numOfHiddenNodes; j++) {
			outputError[t2] = -(target[i] - outputLayer[i]) * outputLayer[i] * (1 - outputLayer[i]) * hiddenLayer[j];
			t2++;
		}
	}

	//calc hidden error adjustment values
	for (int i = 0; i < numOfHiddenNodes; i++) {
		
		float e = 0;
		
		for (int j = 0; j < numOfOutputNodes; j++) {
			e += -(target[j] - outputLayer[j]) * outputLayer[j] * (1 - outputLayer[j]) * outputWeights[i + j * numOfHiddenNodes]; //Might be wrong, needs checking
		}

		for (int j = 0; j < numOfInputNodes; j++) {
			hiddenError[i * numOfInputNodes + j] = e * (hiddenLayer[i] * (1 - hiddenLayer[i])) * inputLayer[j];
		}

	}

	for (int i = 0; i < numOfHiddenNodes*numOfOutputNodes; i++) {
		//update weights output to hidden
		outputWeights[i] = outputWeights[i] - LC * outputError[i];
	}

	for (int i = 0; i < numOfInputNodes*numOfHiddenNodes; i++) {
		//update weights output to hidden
		hiddenWeights[i] = hiddenWeights[i] - LC * hiddenError[i];
	}
}


