defmodule FeedForwardNetwork.BackTest do
  use ExUnit.Case

  alias RingTmp.NeuralNet.Back

  describe "back/3" do
    test "single example pair, single layer network" do
      train_input =
        Matrex.new([
          [0.5],
          [0.5]
        ])

      expected =
        Matrex.new([
          [0],
          [1]
        ])

      weights =
        Matrex.new([
          [0.1, 0.8],
          [0.2, 0.9]
        ])

      bias =
        Matrex.new([
          [0.1],
          [0.2]
        ])

      network = [{:output_layer, weights, bias}]

      [{:output_layer, n_weights, n_bias}] = Back.back([train_input], [expected], network)
      assert n_weights != weights
      assert n_bias != bias
    end

    test "single example pair, two layer network" do
      train_input =
        Matrex.new([
          [0.5],
          [0.5]
        ])

      expected =
        Matrex.new([
          [0],
          [1]
        ])

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

      [{:hidden_layer, nweights_h, nbias_h}, {:output_layer, nweights_o, nbias_o}] =
        Back.back([train_input], [expected], network)

      assert nweights_h != weights_h
      assert nbias_h != bias_h
      assert nweights_o != weights_o
      assert nbias_o != bias_o
    end

    test "two example pair, single layer network" do
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

      weights =
        Matrex.new([
          [0.1, 0.8],
          [0.2, 0.9]
        ])

      bias =
        Matrex.new([
          [0],
          [0]
        ])

      network = [{:output_layer, weights, bias}]

      [{:output_layer, nweights, nbias}] = Back.back(train_inputs, expecteds, network)
      assert nweights != weights
      assert nbias != bias
    end

    test "learning rate can be passed as an option to decrease change to network" do
      train_input =
        Matrex.new([
          [0.5],
          [0.5]
        ])

      expected =
        Matrex.new([
          [0],
          [1]
        ])

      weights =
        Matrex.new([
          [0.1, 0.8],
          [0.2, 0.9]
        ])

      bias =
        Matrex.new([
          [0],
          [0]
        ])

      network = [{:output_layer, weights, bias}]

      [{:output_layer, n_weights, n_bias}] = Back.back([train_input], [expected], network)

      [{:output_layer, lr_weights, lr_bias}] =
        Back.back([train_input], [expected], network, learning_rate: 0.1)

      n_weights_diff = weights |> Matrex.subtract(n_weights)
      lr_weights_diff = weights |> Matrex.subtract(lr_weights)

      assert lr_weights_diff
             |> Matrex.multiply(10)
             |> TestUtils.matrix_equals(n_weights_diff, 0.000001)
    end
  end
end
