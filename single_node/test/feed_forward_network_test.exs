defmodule FeedForwardNetworkTest do
  use ExUnit.Case

  describe "define_network/2" do
    # TODO need to include test to ensure output layer is always last layer
    test "returns a list of layers with initialised parameters" do
      definition = [{:hidden_layer, 4, 6}, {:output_layer, 6, 4}]

      {:reply, {:ok, [{:hidden_layer, h_weights, h_bias}, _o_layer]}, _state} =
        FeedForwardNetwork.handle_call({:define_network, definition, 432}, nil, [])

      assert Matrex.size(h_weights) == {4, 6}
      assert Matrex.size(h_bias) == {6, 1}
    end

    test "returns a list of layer with one item if only one layer in definition" do
      definition = [{:output_layer, 3, 2}]

      {:reply, {:ok, [{:output_layer, o_matrix, o_bias}]}, _state} =
        FeedForwardNetwork.handle_call({:define_network, definition}, nil, [])

      assert Matrex.size(o_matrix) == {3, 2}
      assert Matrex.size(o_bias) == {2, 1}
    end

    test "returns an error if list has no items" do
      definition = []

      assert {:reply, {:error, "empty definition"}, _state} =
               FeedForwardNetwork.handle_call({:define_network, definition}, nil, [])
    end

    test "returns an error is defininitions has incorrect atom" do
      definition = [{:incorrect_layer, 10}]

      assert {:reply, {:error, "{:incorrect_layer, 10} is not a valid definition"}, _state} =
               FeedForwardNetwork.handle_call({:define_network, definition}, nil, [])
    end

    test "returns an error if layer input size is > 0" do
      definition = [{:hidden_layer, 0}]
      definition_m1 = [{:hidden_layer, -1}]

      assert {:reply, {:error, "{:hidden_layer, 0} is not a valid definition"}, _state} =
               FeedForwardNetwork.handle_call({:define_network, definition}, nil, [])

      assert {:reply, {:error, "{:hidden_layer, -1} is not a valid definition"}, _state} =
               FeedForwardNetwork.handle_call({:define_network, definition_m1}, nil, [])
    end

    test "two networks with the same seed have identical initialisations" do
      definition = [{:hidden_layer, 4, 6}, {:hidden_layer, 6, 4}, {:output_layer, 6, 3}]

      {:reply, {:ok, network}, _state} =
        FeedForwardNetwork.handle_call({:define_network, definition, 42}, nil, [])

      {:reply, {:ok, network_2}, _state} =
        FeedForwardNetwork.handle_call({:define_network, definition, 42}, nil, [])

      assert network == network_2
    end
  end

  describe "train/2" do
    test "works as expected" do
      # see related tests in back_test.ex
      train_inputs = [
        Matrex.new([
          [0.5],
          [0.5]
        ]),
        Matrex.new([
          [0.6],
          [0.4]
        ])
      ]

      expecteds = [
        Matrex.new([
          [0],
          [1]
        ]),
        Matrex.new([
          [1],
          [0]
        ])
      ]

      weights_h =
        Matrex.new([
          [0.2, 0.3],
          [0.4, 0.6]
        ])

      bias_h =
        Matrex.new([
          [0.1],
          [0.2]
        ])

      weights_o =
        Matrex.new([
          [0.1, 0.8],
          [0.2, 0.9]
        ])

      bias_o =
        Matrex.new([
          [0],
          [0]
        ])

      network = [{:hidden_layer, weights_h, bias_h}, {:output_layer, weights_o, bias_o}]

      {:reply, {:ok, new_network}, new_network_s} =
        FeedForwardNetwork.handle_call(
          {:train, train_inputs, expecteds, [learning_rate: 0.1]},
          nil,
          network
        )

      assert new_network == new_network_s
    end
  end

  describe "test/2" do
    test "can compute cost correctly for a single layer" do
      weights =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      input =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      expected =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      network = [{:output_layer, weights, bias}]

      {:reply, {:ok, _accuracy, cost}, _state} =
        FeedForwardNetwork.handle_call({:test, [input], [expected]}, nil, network)

      assert Float.round(cost, 5) == 1.03981
    end

    test "can compute accuracy correctly for a single layer" do
      weights =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      input =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      expected =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      network = [{:output_layer, weights, bias}]

      {:reply, {:ok, accuracy, _cost}, _state} =
        FeedForwardNetwork.handle_call({:test, [input], [expected]}, nil, network)

      assert accuracy == 1
    end

    test "can compute loss correctly for a multiple layers" do
      input =
        Matrex.new([
          [0.1],
          [0.2]
        ])

      weights_h =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6]
        ])

      bias_h =
        Matrex.new([
          [0],
          [0.1],
          [0.2]
        ])

      weights_o =
        Matrex.new([
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6]
        ])

      bias_o =
        Matrex.new([
          [0],
          [0]
        ])

      expected =
        Matrex.new([
          [0],
          [1]
        ])

      network = [{:hidden_layer, weights_h, bias_h}, {:output_layer, weights_o, bias_o}]

      {:reply, {:ok, _accuracy, cost}, _state} =
        FeedForwardNetwork.handle_call({:test, [input], [expected]}, nil, network)

      assert Float.round(cost, 5) == 0.66069
    end

    test "can compute accuracy correctly for a multiple layers" do
      input =
        Matrex.new([
          [0.1],
          [0.2]
        ])

      weights_h =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6]
        ])

      bias_h =
        Matrex.new([
          [0],
          [0.1],
          [0.2]
        ])

      weights_o =
        Matrex.new([
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6]
        ])

      bias_o =
        Matrex.new([
          [0],
          [0]
        ])

      expected =
        Matrex.new([
          [0],
          [1]
        ])

      network = [{:hidden_layer, weights_h, bias_h}, {:output_layer, weights_o, bias_o}]

      {:reply, {:ok, accuracy, _cost}, _state} =
        FeedForwardNetwork.handle_call({:test, [input], [expected]}, nil, network)

      assert accuracy == 1
    end

    test "can compute cost for multiple inputs" do
      input_list = [
        Matrex.new([
          [0.1],
          [0.2]
        ]),
        Matrex.new([
          [0.9],
          [0.4]
        ])
      ]

      weights_h =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6]
        ])

      bias_h =
        Matrex.new([
          [0],
          [0.1],
          [0.2]
        ])

      weights_o =
        Matrex.new([
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6]
        ])

      bias_o =
        Matrex.new([
          [0],
          [0]
        ])

      expected_list = [
        Matrex.new([
          [0],
          [1]
        ]),
        Matrex.new([
          [1],
          [0]
        ])
      ]

      network = [{:hidden_layer, weights_h, bias_h}, {:output_layer, weights_o, bias_o}]

      {:reply, {:ok, _accuracy, cost}, _state} =
        FeedForwardNetwork.handle_call({:test, input_list, expected_list}, nil, network)

      assert Float.round(cost, 5) == 0.71421
    end

    test "can compute accuracy for multiple inputs" do
      input_list = [
        Matrex.new([
          [0.1],
          [0.2]
        ]),
        Matrex.new([
          [0.9],
          [0.4]
        ])
      ]

      weights_h =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6]
        ])

      bias_h =
        Matrex.new([
          [0],
          [0.1],
          [0.2]
        ])

      weights_o =
        Matrex.new([
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6]
        ])

      bias_o =
        Matrex.new([
          [0],
          [0]
        ])

      expected_list = [
        Matrex.new([
          [0],
          [1]
        ]),
        Matrex.new([
          [1],
          [0]
        ])
      ]

      network = [{:hidden_layer, weights_h, bias_h}, {:output_layer, weights_o, bias_o}]

      {:reply, {:ok, accuracy, _cost}, _state} =
        FeedForwardNetwork.handle_call({:test, input_list, expected_list}, nil, network)

      assert accuracy == 0.5
    end

    test "will return error if network not initialised" do
      input_list = [1, 2]
      expected_list = [4, 3]
      network = []

      assert {:reply, {:error, "network hasn't been defined"}, network} ==
               FeedForwardNetwork.handle_call({:test, input_list, expected_list}, nil, network)
    end

    test "will return error if test_inputs not the same size as expecteds" do
      input_list = [1, 2]
      expected_list = [1]

      network = [{:hidden_layer, 4, "weights", "bias"}]

      assert {:reply, {:error, "test_input list and expecteds list are not equal"}, network} ==
               FeedForwardNetwork.handle_call({:test, input_list, expected_list}, nil, network)
    end

    # for another time
    # test "will return error if expecteds not one hot encoded" do
    # end

    # test "will return error if test_inputs not the same size as network input" do
    # end
  end

  describe "predict/1" do
    test "prediction with one output layer functions as expected" do
      input_vector =
        Matrex.new([
          [0.3],
          [0.4],
          [0.5]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.7],
          [0.3, 0.4, 0.8],
          [0.5, 0.6, 0.9]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      expected_prediction =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      network = [{:output_layer, weights, bias}]

      {:reply, {:ok, prediction}, _network} =
        FeedForwardNetwork.handle_call({:predict, input_vector}, nil, network)

      assert expected_prediction == prediction
    end

    test "prediction with multiple output layers work as expected" do
      input_vector =
        Matrex.new([
          [0.3],
          [0.4],
          [0.5]
        ])

      weights_h =
        Matrex.new([
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6]
        ])

      bias_h =
        Matrex.new([
          [0.9],
          [0.8]
        ])

      weights_o =
        Matrex.new([
          [0.7, 0.8, 0.9],
          [0.6, 0.5, 0.4]
        ])

      bias_o =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      expected_prediction =
        Matrex.new([
          [1],
          [0],
          [0]
        ])

      network = [{:hidden_layer, weights_h, bias_h}, {:output_layer, weights_o, bias_o}]

      {:reply, {:ok, prediction}, _network} =
        FeedForwardNetwork.handle_call({:predict, input_vector}, nil, network)

      assert expected_prediction == prediction
    end

    # implement this later
    # test "if network hasn't been defined, returns error" do
    #   input_vector =
    #     Matrex.new([
    #       [0.3],
    #       [0.4],
    #       [0.5]
    #     ])

    #   assert {:reply, {:error, "network hasn't been defined yet"}, []} ==
    #            FeedForwardNetwork.handle_call({:predict, input_vector}, nil, [])
    # end

    # TODO implimiment this later
    # test "if input vector is wrong size to enter network" do
    #   input_vector =
    #     Matrex.new([
    #       [0.3],
    #       [0.4],
    #       [0.5]
    #     ])

    #   network = [{:output_layer, "a matrix", "some bias"}]

    #   assert {:reply, {:error, "wrong input length for network, expected 4 given 3"}, network} ==
    #            FeedForwardNetwork.handle_call({:predict, input_vector}, nil, network)
    # end
  end
end
