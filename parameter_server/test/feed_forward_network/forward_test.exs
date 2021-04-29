defmodule FeedForwardNetwork.ForwardTest do
  use ExUnit.Case

  alias FeedForwardNetwork.Forward

  describe "forward_list/2" do
    test "returns a list of activations including the input in reverse order" do
      input =
        Matrex.new([
          [0.3],
          [0.7]
        ])

      weights_1 =
        Matrex.new([
          [0.1, 0.3],
          [0.2, 0.4]
        ])

      bias_1 =
        Matrex.new([
          [0.5],
          [0.6]
        ])

      weights_2 =
        Matrex.new([
          [0.2, 0.6],
          [0.4, 0.8]
        ])

      bias_2 =
        Matrex.new([
          [0.1],
          [0.2]
        ])

      weights_3 =
        Matrex.new([
          [0.9, 0.7],
          [0.8, 0.6]
        ])

      bias_3 = Matrex.new([
        [0.0],
        [0.0]
      ])


      expected =
        [
          Matrex.new([
            [0.3],
            [0.7]
          ]),
          Matrex.new([
            [0.67],
            [0.97]
          ]),
          Matrex.new([
            [0.622],
            [1.378]
          ]),
          Matrex.new([
            [0.59869],
            [0.40131]
          ])
        ]
        |> Enum.reverse()

      expected_z = [
        :na,
        Matrex.new([
          [0.622],
          [1.378]
        ]),
        Matrex.new([
          [0.67],
          [0.97]
        ]),
        :na
      ]

      expected_a = [
        Matrex.new([
          [0.59869],
          [0.40131]
        ]),
        Matrex.new([
          [0.622],
          [1.378]
        ]),
        Matrex.new([
          [0.67],
          [0.97]
        ]),
        Matrex.new([
          [0.3],
          [0.7]
        ])
      ]

      network = [
        {:hidden_layer, weights_1, bias_1},
        {:hidden_layer, weights_2, bias_2},
        {:output_layer, weights_3, bias_3}
      ]

      actual = Forward.forward_list(network, input)

      Enum.map(actual, fn {_z_vec, act_a} -> act_a end)
      |> Enum.zip(expected_a)
      |> Enum.each(fn {act_a, expected_a} ->
        TestUtils.matrix_equals(act_a, expected_a, 0.0001)
      end)

      Enum.map(actual, fn {z_vec, _act} -> z_vec end)
      |> Enum.zip(expected_z)
      |> Enum.each(fn {act_z, expected_z} ->
        case {act_z, expected_z} do
          {:na, :na} -> true
          _ -> TestUtils.matrix_equals(act_z, expected_z, 0.0001)
        end
      end)
    end
  end
end
