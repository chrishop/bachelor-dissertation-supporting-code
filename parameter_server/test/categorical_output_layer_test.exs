defmodule CategoricalOutputLayerTest do
  use ExUnit.Case

  alias CategoricalOutputLayer

  describe "forward/4" do
    test "multiplies matrix and calculates softmax correctly" do
      input =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

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

      expected_output =
        Matrex.new([
          [0.314],
          [0.333],
          [0.354]
        ])

      assert CategoricalOutputLayer.forward(input, weights, bias)
             |> TestUtils.matrix_equals(expected_output, 0.01)
    end

    test "multiplies layers with different input and output sizes" do
      # a 6 value input into a 4 value output
      input =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3],
          [0.4],
          [0.5],
          [0.6]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.3, 0.4],
          [0.5, 0.6, 0.7, 0.8],
          [0.9, 1.0, 1.1, 1.2],
          [1.1, 1.0, 0.9, 0.8],
          [0.7, 0.6, 0.5, 0.4],
          [0.3, 0.2, 0.1, 0.1]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0],
          [0]
        ])

      # correct to 4 dp
      expected =
        Matrex.new([
          [0.2809],
          [0.2567],
          [0.2346],
          [0.2277]
        ])

      assert CategoricalOutputLayer.forward(input, weights, bias)
             |> TestUtils.matrix_equals(expected, 0.0001)
    end
  end

  describe "back/6" do
    test "backpropagation effects weights as expected" do
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
        ])

      activation =
        Matrex.new([
          [0.314],
          [0.333],
          [0.354]
        ])

      target_activation =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      # to 3 decimal places
      expected_delta =
        Matrex.new([
          [-0.0314, -0.0333, 0.0646],
          [-0.0628, -0.0666, 0.1292],
          [-0.0942, -0.0999, 0.1938]
        ])

      expected_new_weights = Matrex.add(weights, expected_delta)

      {new_weights, _, _remaining_error} =
        CategoricalOutputLayer.back(activation, prev_activation, weights, bias, target_activation)

      assert TestUtils.matrix_equals(new_weights, expected_new_weights, 0.0001)
    end

    test "learning rates effect new weights as expected" do
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
        ])

      activation =
        Matrex.new([
          [0.314],
          [0.333],
          [0.354]
        ])

      target_activation =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      opts = [learning_rate: 0.1]

      # to 4 decimal places
      expected_delta =
        Matrex.new([
          [-0.0314, -0.0333, 0.0646],
          [-0.0628, -0.0666, 0.1292],
          [-0.0942, -0.0999, 0.1938]
        ])

      expected_new_weights = Matrex.multiply(expected_delta, 0.1) |> Matrex.add(weights)

      {new_weights, _, _remaining_error} =
        CategoricalOutputLayer.back(
          activation,
          prev_activation,
          weights,
          bias,
          target_activation,
          opts
        )

      assert TestUtils.matrix_equals(new_weights, expected_new_weights, 0.0001)
    end

    test "remaining error is calculated as expected" do
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
        ])

      activation =
        Matrex.new([
          [0.314],
          [0.333],
          [0.354]
        ])

      target_activation =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      # to 3 decimal places
      expected_error =
        Matrex.new([
          [0.0958],
          [0.0955],
          [0.0952]
        ])

      {_new_weights, _, remaining_error} =
        CategoricalOutputLayer.back(activation, prev_activation, weights, bias, target_activation)

      assert TestUtils.matrix_equals(remaining_error, expected_error, 0.0001)
    end

    test "can calculate of a layer with a different input size and output size" do
      # input layer of 6 output layer of 4
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3],
          [0.4],
          [0.5],
          [0.6]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.3, 0.4],
          [0.5, 0.6, 0.7, 0.8],
          [0.9, 1.0, 1.1, 1.2],
          [1.1, 1.0, 0.9, 0.8],
          [0.7, 0.6, 0.5, 0.4],
          [0.3, 0.2, 0.1, 0.1]
        ])

      # correct to 4 dp
      activation =
        Matrex.new([
          [0.2809],
          [0.2567],
          [0.2346],
          [0.2277]
        ])

      target_activation =
        Matrex.new([
          [1],
          [0],
          [0],
          [0]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0],
          [0]
        ])

      {new_weights, _, remaining_error} =
        CategoricalOutputLayer.back(
          activation,
          prev_activation,
          weights,
          bias,
          target_activation,
          []
        )

      assert Matrex.size(new_weights) == {6, 4}
      assert Matrex.size(remaining_error) == {6, 1}
    end
  end

  describe "loss/2" do
    test "it produces expected values" do
      activation =
        Matrex.new([
          [0.314],
          [0.333],
          [0.354]
        ])

      target_activation =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      # correct to 5 decimal places
      expected_loss = 1.03846

      assert expected_loss ==
               Float.round(CategoricalOutputLayer.loss(activation, target_activation), 5)
    end
  end

  describe "after each iteration the loss should decrease" do
    test "two iteratations" do
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      weights =
        Matrex.new([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
        ])

      activation =
        Matrex.new([
          [0.314],
          [0.333],
          [0.354]
        ])

      target_activation =
        Matrex.new([
          [0],
          [0],
          [1]
        ])

      bias =
        Matrex.new([
          [0],
          [0],
          [0]
        ])

      loss_0 = CategoricalOutputLayer.loss(activation, target_activation)

      {new_weights, _, _remaining_error} =
        CategoricalOutputLayer.back(activation, prev_activation, weights, bias, target_activation,
          learning_rate: 0.1
        )

      new_activation = CategoricalOutputLayer.forward(prev_activation, new_weights, bias)
      loss_1 = CategoricalOutputLayer.loss(new_activation, target_activation)

      {new_weights, new_bias, _remaining_error} =
        CategoricalOutputLayer.back(
          new_activation,
          prev_activation,
          new_weights,
          bias,
          target_activation,
          learning_rate: 0.1
        )

      new_activation = CategoricalOutputLayer.forward(prev_activation, new_weights, new_bias)
      loss_2 = CategoricalOutputLayer.loss(new_activation, target_activation)

      {new_weights, new_bias, _remaining_error} =
        CategoricalOutputLayer.back(
          new_activation,
          prev_activation,
          new_weights,
          bias,
          target_activation,
          learning_rate: 0.1
        )

      new_activation = CategoricalOutputLayer.forward(prev_activation, new_weights, new_bias)
      loss_3 = CategoricalOutputLayer.loss(new_activation, target_activation)

      assert loss_0 > loss_1 and loss_1 > loss_2 and loss_2 > loss_3
    end
  end
end
