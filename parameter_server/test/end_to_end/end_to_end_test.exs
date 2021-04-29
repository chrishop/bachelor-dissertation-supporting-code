defmodule EndToEndTest do
  use ExUnit.Case

  @moduletag end_to_end: true

  describe "Iris dataset" do
    test "single layer neural network, single epoch" do
      # get training and testing data
      {train_set, test_set} = Iris.train_test_data(0.8)

      {train_inputs, train_examples} = Enum.unzip(train_set)
      {test_inputs, test_examples} = Enum.unzip(test_set)

      # start link with neural network
      FeedForwardNetwork.start_link([])

      {:ok, network} = FeedForwardNetwork.define_network([{:output_layer, 4, 3}])
      {:ok, init_accuracy, init_cost} = FeedForwardNetwork.test(test_inputs, test_examples)

      {:ok, new_network} = FeedForwardNetwork.train(train_inputs, train_examples, learning_rate: 0.1)

      # {:ok, accuracy, cost} = FeedForwardNetwork.test(train_inputs, train_examples)
      {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(test_inputs, test_examples)

      [{:output_layer, init_matrix, init_bias}] = network

      [{:output_layer, trained_matrix, trained_bias}] = new_network

      assert init_accuracy < train_accuracy
      assert init_cost > train_cost
    end

    test "hidden layer and output layer, single epoch" do
      {train_set, test_set} = Iris.train_test_data(0.8)

      {train_inputs, train_examples} = Enum.unzip(train_set)
      {test_inputs, test_examples} = Enum.unzip(test_set)

      # train_inputs = Enum.take(train_inputs, 5)
      # train_examples = Enum.take(train_examples, 5)

      # start link with neural network
      FeedForwardNetwork.start_link([])

      {:ok, network} = FeedForwardNetwork.define_network([{:hidden_layer, 4, 5}, {:output_layer, 5, 3}])
      {:ok, init_accuracy, init_cost} = FeedForwardNetwork.test(test_inputs, test_examples)

      {:ok, new_network} = FeedForwardNetwork.train(train_inputs, train_examples, learning_rate: 0.01)

      {:ok, accuracy, cost} = FeedForwardNetwork.test(train_inputs, train_examples)
      {:ok, test_accuracy, test_cost} = FeedForwardNetwork.test(test_inputs, test_examples)
      {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(train_inputs, train_examples)

      [
        {:hidden_layer, init_weights_h, init_bias_h},
        {:output_layer, init_weights_o, init_bias_o}
      ] = network

      [
        {:hidden_layer, trained_weights_h, trained_bias_h},
        {:output_layer, trained_weights_o, trained_bias_o}
      ] = new_network

      IO.puts("hidden_weights change")
      IO.inspect(init_weights_h)
      IO.inspect(trained_weights_h)
      Matrex.heatmap(init_weights_h)
      Matrex.heatmap(trained_weights_h)

      IO.puts("hidden_bias change")
      IO.inspect(init_bias_h)
      IO.inspect(trained_bias_h)
      # Matrex.heatmap(init_bias_h)
      # Matrex.heatmap(trained_bias_h)

      IO.puts("output_weights change")
      IO.inspect(init_weights_o)
      IO.inspect(trained_weights_o)
      Matrex.heatmap(init_weights_o)
      Matrex.heatmap(trained_weights_o)


      IO.puts("output_bias change")
      IO.inspect(init_bias_o)
      IO.inspect(trained_bias_o)
      # Matrex.heatmap(init_bias_o)
      # Matrex.heatmap(trained_bias_o)


      IO.inspect("Results")
      IO.inspect("init_accuracy: #{init_accuracy}, test_data_accuracy: #{test_accuracy}, train_data_accuracy: #{train_accuracy}")
      IO.inspect("init_cost: #{init_cost}, test_data_cost: #{test_cost}, train_data_cost: #{train_cost}")

      assert init_weights_h != trained_weights_h
      assert init_weights_o != trained_weights_o

      assert init_accuracy < train_accuracy
      assert init_cost > train_cost
    end
  end
end
