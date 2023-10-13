/*
 * NAME: John Virga
 * CSC 475 AI
 * Assignment 2 P1
 * 
 */

import java.lang.Math;


public class MyNeuralNetwork{
	@SuppressWarnings("unused")
	public static void main(String[] args) {
		
		// What nodes and where
		
		int inputLayerNodes = 4;
		int hiddenLayerNodes = 3;
		int outputLayerNodes = 2;
		
		
		// Constants
		int learningRate = 10;
		
		// Training Data: 
		// Input Layers & Expectant Output Layers
		double[] inputLayer1 = {0,1,0,1};
		double[] outputLayer1 = {0,1};
		
		double[] inputLayer2 = {1,0,1,0};
		double[] outputLayer2 = {1,0};
		
		double[] inputLayer3 = {0,0,1,1};
		double[] outputLayer3 = {0,1};
		
		double[] inputLayer4 = {1,1,0,0};
		double[] outputLayer4 = {1,0};
		
		// Pseudo random values
		// Weights and biases given for assignment separated by layer
		
		// Layer 0: Input Layer
		double[] initialWeightsLayer0 = {
				-0.21, 0.72, -0.25, 1,
				-0.94, -0.41, -0.47, 0.63,
				0.15, 0.55, -0.49, -0.75
				};

		double[] initialBiasesLayer0 = {
				0.1,
				-0.36,
				-0.31
		};
		
		// Layer 1: Hidden Layer
		double[] initialWeightsLayer1 = {
				0.76, 0.48, -0.73,
				0.34, 0.89, -0.23
				};

		double[] initialBiasesLayer1 = {
				0.16,
				-0.46
		};
		
		// Command Run
		/*
		Find our z value to plug into sigmoidal function
		 - sum up the W_k * a_k + b_k where W is weight at k and a_k is input at k and b is the bias at k
		Send z value to sigmoid fn
		
		 */
		
		double[] z = getZ(inputLayer1, initialWeightsLayer0, initialBiasesLayer0, inputLayerNodes, hiddenLayerNodes);
		
		// Print Z values 
		
//		System.out.println("z values: ");
//		for(double element : z) {
//			System.out.println(element);
//		}
//		System.out.println();
		
		
		double[] sigmoids = getSigmoid(z);
		
		//Print sigmoid values
		
//		System.out.println("Sigmoid values: ");
//		for(double element : sigmoids) {
//			System.out.println(element);
//		}
		
	}
	
	// Both getZ and getSigmoid are part of the forward pass between layers
	public static double[] getZ(double[] input, double[] weight, double[] bias, int nodesInCurrentLayer, int nodesInNextLayer) {
		
		// Single column array whose length is equal to nodes in the next layer
		double z[] = new double[nodesInNextLayer];
		
		for(int i=0;i<nodesInCurrentLayer-1;i++) {
			for(int j=0;j<nodesInCurrentLayer;j++) {
				z[i] += weight[j+i*nodesInCurrentLayer]*input[j];
			}
		}
			
		// Add bias to each row of z
		for(int i=0; i<z.length; i++) {
			z[i] += bias[i];
		}
		
		return z;
	}
	
	public static double[] getSigmoid(double[] z) {
		
		// Iterating through z array, we can substitute each previous value of z with the sigmoid fn given z
		for(int i=0; i<z.length; i++) {
			z[i] = 1/(1+Math.pow(2.71828,-z[i]));
		}
		
		// Returns the activation value of next layer
		return z;
	}
	
	
	// Backward Pass
	public static double[] name() {
		
		return null;
	}
	
}
