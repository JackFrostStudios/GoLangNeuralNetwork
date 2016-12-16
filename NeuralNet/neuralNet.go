package NeuralNet

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

//SetUpNodes will create a Network objects with the requested number of inputs, hidden nodes and outputs
func SetUpNodes(inputs int, hiddenNodes int, outputs int) Network {
	var hiddenLayerNodes []Node
	var outputNodes []Node

	for i := 0; i < hiddenNodes; i++ {
		var newNode Node
		newNode.NetworkPositionID = fmt.Sprint("h", i)
		for j := 0; j < inputs; j++ {
			var weight InputWeight
			weight.InputNodeID = fmt.Sprint("i", j)
			weight.Weight = rand.Float64()
			newNode.Weights = append(newNode.Weights, weight)
		}
		newNode.BiasWeight = rand.Float64()
		hiddenLayerNodes = append(hiddenLayerNodes, newNode)
	}

	for i := 0; i < outputs; i++ {
		var newNode Node
		newNode.NetworkPositionID = fmt.Sprint("o", i)
		for j := 0; j < hiddenNodes; j++ {
			var weight InputWeight
			weight.InputNodeID = hiddenLayerNodes[j].NetworkPositionID
			weight.Weight = rand.Float64()
			newNode.Weights = append(newNode.Weights, weight)
		}
		newNode.BiasWeight = rand.Float64()
		outputNodes = append(outputNodes, newNode)
	}
	return Network{Nodes: hiddenLayerNodes, OutputNodes: outputNodes}
}

//Network struct that contains the hidden nodes and output nodes
type Network struct {
	Iteration       int
	Inputs          []float64
	ExpectedOutputs []float64
	Nodes           []Node
	OutputNodes     []Node
}

//CalculateOutputs calculates the value for each hidden node based on the input, then calculate the output nodes values based on result.
func (n *Network) CalculateOutputs(inputs []float64) {
	var nodeResults []float64
	for index := range n.Nodes {
		n.Nodes[index].calculate(inputs)
		nodeResults = append(nodeResults, n.Nodes[index].Output)
	}

	for index := range n.OutputNodes {
		n.OutputNodes[index].calculate(nodeResults)
	}
}

//CalculateError calculates the cost function and then uses this to update the weights of the output nodes and then back propegate to each hidden node.
func (n *Network) CalculateError(inputs []float64, expectedOutputs []float64) {
	var outputErrorResults []float64
	//Calculate Output nodes error vs expected result
	for index, outputNode := range n.OutputNodes {
		errorResult := outputNode.Output * (1 - outputNode.Output) * (expectedOutputs[index] - outputNode.Output)
		outputErrorResults = append(outputErrorResults, errorResult)
		n.OutputNodes[index].ErrorGradient = errorResult
	}

	//Calculate hidden nodes error based on Output node error times weight
	for hiddenIndex, hiddenNode := range n.Nodes {
		var weightedError float64
		weightedError = 0

		for outputIndex, outputNode := range n.OutputNodes {
			weightedError += outputErrorResults[outputIndex] * outputNode.Weights[hiddenIndex].Weight
		}

		errorResult := hiddenNode.Output * (1 - hiddenNode.Output) * weightedError
		n.Nodes[hiddenIndex].ErrorGradient = errorResult
	}
}

//UpdateWeightsBasedOnError will update each node in the network based on the current node input and errorGradient
func (n *Network) UpdateWeightsBasedOnError() {
	for index := range n.OutputNodes {
		n.OutputNodes[index].updateWeightBasedOnError()
	}
	for index := range n.Nodes {
		n.Nodes[index].updateWeightBasedOnError()
	}
}

//Clone will return a deep copy of a network
func (n *Network) Clone() Network {
	var returnNetwork Network
	for _, outputNode := range n.OutputNodes {
		returnNetwork.OutputNodes = append(returnNetwork.OutputNodes, outputNode.clone())
	}
	for _, node := range n.Nodes {
		returnNetwork.Nodes = append(returnNetwork.Nodes, node.clone())
	}
	returnNetwork.Iteration = n.Iteration
	for _, input := range n.Inputs {
		returnNetwork.Inputs = append(returnNetwork.Inputs, input)
	}
	for _, expectedOutputs := range n.ExpectedOutputs {
		returnNetwork.ExpectedOutputs = append(returnNetwork.ExpectedOutputs, expectedOutputs)
	}

	return returnNetwork
}

//ClearTrainingData will remove all input / outputs that relate to a specific training step
func (n *Network) ClearTrainingData() {
	n.Iteration = 0
	n.Inputs = nil
	n.ExpectedOutputs = nil
	for i := range n.Nodes {
		n.Nodes[i].clearTrainingData()
	}
	for i := range n.OutputNodes {
		n.OutputNodes[i].clearTrainingData()
	}
}

//Node represents a network node
type Node struct {
	NetworkPositionID string
	Weights           []InputWeight
	BiasWeight        float64
	Inputs            []float64
	Output            float64
	ErrorGradient     float64
}

//InputWeight represents the weight assigned to a specific input
type InputWeight struct {
	Weight      float64
	InputNodeID string
}

func (n *Node) calculate(inputs []float64) (err error) {
	if len(inputs) != len(n.Weights) {
		return errors.New("Inputs length does not match weights")
	}
	n.Inputs = inputs
	var total float64
	total = 0
	for index, input := range inputs {
		total += (input * n.Weights[index].Weight)
	}
	total += n.BiasWeight * -1
	total = 1.0 / (1.0 + math.Exp(-total))
	n.Output = total
	return
}

func (n *Node) updateWeightBasedOnError() {
	for index := range n.Weights {
		n.Weights[index].Weight += 1 * n.Inputs[index] * n.ErrorGradient
	}
	n.BiasWeight += 1 * -1 * n.ErrorGradient
}

func (n *Node) clone() Node {
	var clone Node
	clone.NetworkPositionID = n.NetworkPositionID
	clone.BiasWeight = n.BiasWeight
	clone.Output = n.Output
	clone.ErrorGradient = n.ErrorGradient
	for _, weight := range n.Weights {
		var clonedWeight InputWeight
		clonedWeight.Weight = weight.Weight
		clonedWeight.InputNodeID = weight.InputNodeID
		clone.Weights = append(clone.Weights, clonedWeight)
	}
	for _, input := range n.Inputs {
		clone.Inputs = append(clone.Inputs, input)
	}
	return clone
}

func (n *Node) clearTrainingData() {
	n.Inputs = nil
	n.Output = 0
	n.ErrorGradient = 0
}
