import java.lang.Math;


public class MainClass {
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
		// Weights given for assignment
		double[] initialWeights = {
				-0.21, 0.72, -0.25, 1,
				-0.94, -0.41, -0.47, 0.63,
				0.15, 0.55, -0.49, -0.75
				};
		// Biases given for assignment
		double[] initialBiases = {
				0.1,
				-0.36,
				-0.31
		};
		
		// Command Run
		/*
		Find our z value to plug into sigmoidal function
		 - sum up the W_k * a_k + b_k where W is weight at k and a_k is input at k and b is the bias at k
		Send z value to sigmoid fn
		
		 */
		
		double[] z = getZ(inputLayer1, initialWeights, initialBiases, inputLayerNodes, hiddenLayerNodes);
		
		System.out.println("z values: ");
		for(double element : z) {
			System.out.println(element);
		}
		System.out.println();
		
		double[] sigmoids = getSigmoid(z);
		
		System.out.println("Sigmoid values: ");
		for(double element : sigmoids) {
			System.out.println(element);
		}
	}
	
	public static double[] getZ(double[] input, double[] weight, double[] bias, int nodesInCurrentLayer, int nodesInNextLayer) {
		
		// Single column array whose length is equal to nodes in the next layer
		double z[] = new double[nodesInNextLayer];
		
		// Matrix math implementation that populates z with the sum of each weight 
		// in row 'weight' multiplied by singular corresponding input of the same row.
		for(int inputPosition=0; inputPosition<nodesInCurrentLayer-1; inputPosition++){
			for(int weightPosition=0; weightPosition<nodesInNextLayer-1; weightPosition++) {
				z[inputPosition] += weight[weightPosition]*input[inputPosition]; 
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
	
	
}
