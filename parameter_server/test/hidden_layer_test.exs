defmodule HiddenLayerTest do
  use ExUnit.Case

  alias HiddenLayer

  @prev_activation Matrex.new([
                     [0.2],
                     [0.5],
                     [0.3]
                   ])

  @weights Matrex.new([
             [1.5, 2, 2.5],
             [-1, 0.2, -1],
             [1.2, 1, 0.5]
           ])

  @bias Matrex.new([
          [-0.06],
          [-0.6],
          [0.15]
        ])

  @remaining_error Matrex.new([
                     [0.2876],
                     [0.2875],
                     [0.2874]
                   ])

  describe "forward/4" do
    test "multiplies matrix and applies relu function correctly" do
      expected_z_vec =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      expected_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      {z_vec, activation} = HiddenLayer.forward(@prev_activation, @weights, @bias)

      assert z_vec == activation
      assert TestUtils.matrix_equals(z_vec, expected_z_vec, 0.00001)
      assert TestUtils.matrix_equals(activation, expected_activation, 0.0000001)
    end

    test "negative z_vec gives 0'd activation" do
      bias =
        Matrex.new([
          [-0.26],
          [-0.6],
          [0.15]
        ])

      expected_z_vec =
        Matrex.new([
          [-0.1],
          [0.2],
          [0.3]
        ])

      expected_activation =
        Matrex.new([
          [0],
          [0.2],
          [0.3]
        ])

      {z_vec, activation} = HiddenLayer.forward(@prev_activation, @weights, bias)

      assert z_vec != activation
      assert TestUtils.matrix_equals(z_vec, expected_z_vec, 0.00001)
      assert TestUtils.matrix_equals(activation, expected_activation, 0.0000001)
    end

    test "works with layers of different dimensions" do
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3],
          [0.4]
        ])

      weights =
        Matrex.new([
          [0.1, 0.5, 0.9, 1.1, 0.7, 0.3],
          [0.2, 0.6, 1.0, 1.0, 0.6, 0.2],
          [0.3, 0.7, 1.1, 0.9, 0.5, 0.1],
          [0.4, 0.8, 1.2, 0.8, 0.4, 0.1]
        ])

      bias =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3],
          [0.4],
          [0.5],
          [0.6]
        ])

      {z_vec, activation} = HiddenLayer.forward(prev_activation, weights, bias)

      assert z_vec |> Matrex.size() == {6, 1}
      assert activation |> Matrex.size() == {6, 1}
    end
  end

  describe "back/6" do
    # TODO Figure out if I need to fix this test
    # test "new weight is calculated correctly" do
    #   z_vec =
    #     Matrex.new([
    #       [0.1],
    #       [0.2],
    #       [0.3]
    #     ])

    #   # expected_weight_delta =
    #   #   Matrex.new([
    #   #     [0.17256, 0.1725, 0.17244],
    #   #     [0.4314, 0.43125, 0.4311],
    #   #     [0.25884, 0.25875, 0.25866]
    #   #   ])

    #   # expected_weight_delta =
    #   #   Matrex.new([
    #   #     [0.05752, 0.0575, 0.05748],
    #   #     [0.1438, 0.14375, 0.1437],
    #   #     [0.08628, 0.08625, 0.08622]
    #   #   ])

    #   expected_weights = Matrex.add(@weights, expected_weight_delta)

    #   {new_weights, _new_bias, _remaining_error} =
    #     HiddenLayer.back(z_vec, @prev_activation, @weights, @bias, @remaining_error)

    #   assert TestUtils.matrix_equals(new_weights, expected_weights, 0.00001)
    # end

    test "new_bias is calculated correctly" do
      z_vec =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      expected_bias_delta =
        Matrex.new([
          [0.2876],
          [0.2875],
          [0.2874]
        ])

      {_new_weights, new_bias, _layer_error} =
        HiddenLayer.back(z_vec, @prev_activation, @weights, @bias, @remaining_error)

      expected_bias = Matrex.add(@bias, expected_bias_delta)

      assert TestUtils.matrix_equals(new_bias, expected_bias, 0.000001)
    end

    test "remaining_error is calculated" do
      z_vec =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3]
        ])

      expected_remaining_error =
        Matrex.new([
          [1.7249],
          [-0.5175],
          [0.77632]
        ])

      {_new_weights, _new_bias, remaining_error} =
        HiddenLayer.back(z_vec, @prev_activation, @weights, @bias, @remaining_error)

      assert TestUtils.matrix_equals(remaining_error, expected_remaining_error, 0.0001)
    end

    test "works with different dimensions" do
      # a layer with input of 4 and output of 6
      prev_activation =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3],
          [0.4]
        ])

      weights =
        Matrex.new([
          [0.1, 0.5, 0.9, 1.1, 0.7, 0.3],
          [0.2, 0.6, 1.0, 1.0, 0.6, 0.2],
          [0.3, 0.7, 1.1, 0.9, 0.5, 0.1],
          [0.4, 0.8, 1.2, 0.8, 0.4, 0.1]
        ])

      bias =
        Matrex.new([
          [0.1],
          [0.2],
          [0.3],
          [0.4],
          [0.5],
          [0.6]
        ])

      z_vec =
        Matrex.new([
          [0.4],
          [0.9],
          [1.4],
          [1.3],
          [1.0],
          [0.74]
        ])

      remaining_error =
        Matrex.new([
          [0.098],
          [0.095],
          [0.084],
          [0.076],
          [0.053],
          [0.069]
        ])

      {new_weights, new_bias, prev_remaining_error} =
        HiddenLayer.back(z_vec, prev_activation, weights, bias, remaining_error)

      assert Matrex.size(new_weights) == {4, 6}
      assert Matrex.size(new_bias) == {6, 1}
      assert Matrex.size(prev_remaining_error) == {4, 1}
    end
  end
end
