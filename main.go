package main

import (
	"NeuralNetwork/NeuralNet"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	_ "github.com/denisenkom/go-mssqldb"
)

func main() {
	//inputsSize, hiddenNodesSize, outputsSize := getNetworkStructure()
	var hiddenNodes int
	fmt.Printf("How many nodes would you like?\n")
	fmt.Scanln(&hiddenNodes)
	neuralNetwork := NeuralNet.SetUpNodes(3, hiddenNodes, 1)
	var initialWeights []float64
	for _, outputNode := range neuralNetwork.OutputNodes {
		for _, weight := range outputNode.Weights {
			initialWeights = append(initialWeights, weight.Weight)
		}
	}

	db, err := sql.Open("mssql", "server=LAPTOP-REEVESJ;user id=sa;password=;database=neuralNetworkInput")
	defer db.Close()
	if err != nil {
		log.Fatal(err)
		fmt.Scanln()
		return
	}

	result, err := db.Query("SELECT Input1, Input2, Input3, Output FROM [dbo].[Training_Cases]")
	if err != nil {
		log.Fatal(err)
		fmt.Scanln()
		return
	}
	var i float64
	var input1, input2, input3, output float64
	var networksOutput []NeuralNet.Network
	//for result.Next() {
	//i++
	for i := 0; i < 10; i++ {
		result.Next()
		networksOutput = append(networksOutput, neuralNetwork.Clone())
		fmt.Printf("------ Training with Input %d ------\n", int(i))
		if err := result.Scan(&input1, &input2, &input3, &output); err != nil {
			log.Fatal(err)
			fmt.Scanln()
			return
		}
		inputs := []float64{input1, input2, input3}
		expectedOutput := []float64{output}
		neuralNetwork.CalculateOutputs(inputs)
		neuralNetwork.CalculateError(inputs, expectedOutput)
		neuralNetwork.UpdateWeightsBasedOnError()
	}
	fmt.Printf("FINAL EXPECTED OUTPUT %f\n", output)
	for _, outputNode := range neuralNetwork.OutputNodes {
		fmt.Printf("ACTUAL OUTPUT: %f\n", outputNode.Output)
	}

	result, err = db.Query("SELECT Input1, Input2, Input3, Output FROM [dbo].[TestCases]")
	if err != nil {
		log.Fatal(err)
		fmt.Scanln()
		return
	}

	i = 0
	var maxDiffAbove float64
	var maxDiffBelow float64
	var averageDiff float64
	for result.Next() {
		i++
		fmt.Printf("------ Test %d ------\n", int(i))
		if err := result.Scan(&input1, &input2, &input3, &output); err != nil {
			log.Fatal(err)
			fmt.Scanln()
			return
		}
		inputs := []float64{input1, input2, input3}
		neuralNetwork.CalculateOutputs(inputs)
		fmt.Printf("EXPECTED OUTPUT %f\n", output*100)
		for _, outputNode := range neuralNetwork.OutputNodes {
			diff := (output - outputNode.Output) * 100
			if diff > maxDiffAbove {
				maxDiffAbove = diff
			}
			if diff < maxDiffBelow {
				maxDiffBelow = diff
			}

			if diff < 0 {
				averageDiff += diff * -1
			} else {
				averageDiff += diff
			}
			fmt.Printf("ACTUAL OUTPUT: %f\n", outputNode.Output*100)
			fmt.Printf("Difference %f\n", diff)
		}
	}

	averageDiff = averageDiff / i
	fmt.Printf("\n------Totals--------\n")
	fmt.Printf("Maximum Overestimate: %f\n", maxDiffAbove)
	fmt.Printf("Maximum Underestimate: %f\n", maxDiffBelow)
	fmt.Printf("Average Error: %f\n", averageDiff)

	fmt.Printf("\n------Final Weights--------\n")

	for _, outputNode := range neuralNetwork.OutputNodes {
		for index, weight := range outputNode.Weights {

			fmt.Printf("----- Weight Change for node %d -----\n", index+1)
			fmt.Printf("Initial Weight: %f\n", initialWeights[index])

			fmt.Printf("Final Weight: %f\n", weight.Weight)
		}
	}

	for index, net := range networksOutput {

		fmt.Printf("----- Netowrk %d -----\n", index+1)
		output, _ := json.MarshalIndent(net, "", " ")
		fmt.Println(string(output))
	}

	fmt.Scanln()
	return

}

func getNetworkStructure() (numberOfInputs int, numberOfHiddenNodes int, numberOfOutputs int) {
	fmt.Printf("How many inputs would you like?")
	fmt.Scanln(&numberOfInputs)
	fmt.Printf("How many hidden nodes would you like?")
	fmt.Scanln(&numberOfHiddenNodes)
	fmt.Printf("How many outputs would you like?")
	fmt.Scanln(&numberOfOutputs)
	return
}
