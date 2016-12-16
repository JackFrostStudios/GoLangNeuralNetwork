package main

import (
	"GoLangNeuralNetwork/NeuralNet"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	_ "github.com/denisenkom/go-mssqldb"
)

func main() {
	//Set Up initial network
	var hiddenNodes int
	fmt.Printf("How many nodes would you like?\n")
	fmt.Scanln(&hiddenNodes)
	neuralNetwork := NeuralNet.SetUpNodes(3, hiddenNodes, 1)
	var initialNetwork NeuralNet.Network
	initialNetwork = neuralNetwork.Clone()
	initialNetwork.ClearTrainingData()

	//Connect with Db
	db, err := sql.Open("mssql", "server=LAPTOP-REEVESJ;user id=sa;password=;database=neuralNetworkInput")
	defer db.Close()
	if err != nil {
		log.Fatal(err)
		fmt.Scanln()
		return
	}
	//Train Network
	trainingIterations, finalNetwork := trainNetwork(neuralNetwork, db)

	//Output results from training
	fmt.Printf("----- Final Training Results -----\n")
	outputNum := 1
	for _, net := range trainingIterations {
		if outputNum == 250 {
			fmt.Printf("----- Training Iteration %d -----\n", net.Iteration)
			output, _ := json.MarshalIndent(net, "", " ")
			fmt.Println(string(output))
			outputNum = 1
		} else {
			outputNum++
		}
	}

	fmt.Printf("----- Initial Network -----\n")
	initialNetworkOutput, _ := json.MarshalIndent(initialNetwork, "", " ")
	fmt.Println(string(initialNetworkOutput))

	fmt.Printf("----- Final Network -----\n")
	finalNetworkOutput, _ := json.MarshalIndent(finalNetwork, "", " ")
	fmt.Println(string(finalNetworkOutput))

	//Test Network
	maxDiffAbove, maxDiffBelow, averageDiff := testNetwork(finalNetwork.Clone(), db)

	fmt.Printf("\n------Totals--------\n")
	fmt.Printf("Maximum Overestimate: %f\n", maxDiffAbove)
	fmt.Printf("Maximum Underestimate: %f\n", maxDiffBelow)
	fmt.Printf("Average Error: %f\n", averageDiff)

	fmt.Scanln()
	return

}

func trainNetwork(neuralNetwork NeuralNet.Network, db *sql.DB) (trainingIterations []NeuralNet.Network, finalNetwork NeuralNet.Network) {
	fmt.Printf("------ Training Network ------\n")

	result, err := db.Query("SELECT Input1, Input2, Input3, Output FROM [dbo].[Training_Cases]")
	if err != nil {
		log.Fatal(err)
		fmt.Scanln()
		return
	}

	var input1, input2, input3, output float64
	var iterations []NeuralNet.Network

	i := 0
	for result.Next() {
		i++

		if err := result.Scan(&input1, &input2, &input3, &output); err != nil {
			log.Fatal(err)
			fmt.Scanln()
			return
		}
		inputs := []float64{input1, input2, input3}
		expectedOutput := []float64{output}

		neuralNetwork.CalculateOutputs(inputs)
		neuralNetwork.CalculateError(inputs, expectedOutput)

		networkToSave := neuralNetwork.Clone()
		networkToSave.Iteration = i
		networkToSave.Inputs = inputs
		networkToSave.ExpectedOutputs = expectedOutput
		iterations = append(iterations, networkToSave)

		neuralNetwork.UpdateWeightsBasedOnError()
	}
	trainingIterations = iterations
	finalNetwork = neuralNetwork.Clone()
	finalNetwork.ClearTrainingData()
	return
}

func testNetwork(neuralNetwork NeuralNet.Network, db *sql.DB) (maxDiffAbove float64, maxDiffBelow float64, averageDiff float64) {
	fmt.Printf("------ Testing Network ------\n")

	result, err := db.Query("SELECT Input1, Input2, Input3, Output FROM [dbo].[TestCases]")
	if err != nil {
		log.Fatal(err)
		fmt.Scanln()
		return
	}

	var input1, input2, input3, output float64
	i := 0

	for result.Next() {
		i++
		if err := result.Scan(&input1, &input2, &input3, &output); err != nil {
			log.Fatal(err)
			fmt.Scanln()
			return
		}
		inputs := []float64{input1, input2, input3}
		neuralNetwork.CalculateOutputs(inputs)
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
		}
	}

	averageDiff = averageDiff / float64(i)

	return
}
