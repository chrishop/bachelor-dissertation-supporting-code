defmodule FeedForwardNetwork.Test do
  alias FeedForwardNetwork.Predict
  alias FeedForwardNetwork.Forward

  def test(test_inputs, expecteds, network) do
    case validate_test_inputs(test_inputs, expecteds, network) do
      :ok ->
        do_tests(test_inputs, expecteds, network)
        |> (fn {accuracy, cost} -> {:ok, accuracy, cost} end).()

      {:error, msg} ->
        {:error, msg}
    end
  end

  defp do_tests(test_inputs, expecteds, network) do
    zipped = Enum.zip(test_inputs, expecteds)

    accuracy =
      zipped
      |> Enum.reduce(
        0,
        fn {input, expected}, acc ->
          case Predict.prediction(input, network) do
            {:ok, prediction} when prediction == expected ->
              1 + acc

            {:ok, _prediction} ->
              acc

            {:error, _reason} ->
              acc
          end
        end
      )
      |> Kernel./(length(test_inputs))

    cost =
      zipped
      |> Enum.reduce(
        0,
        fn {input, expected}, acc ->
          Forward.forward(network, input)
          |> CategoricalOutputLayer.loss(expected)
          |> Kernel.+(acc)
        end
      )
      |> Kernel./(length(test_inputs))

    {accuracy, cost}
  end

  defp validate_test_inputs(test_inputs, expecteds, network) do
    # hot_encoded = is_one_hot_encoded(expecteds)

    # need to put this in a reduce
    # {_layer, input_size, _weights, _bias} = List.first(network) || {nil, 0, nil, nil}
    # {column, _row} = Matrex.size(test_inputs)
    # same_size = input_size == column

    # could validate network at the same time

    case {length(test_inputs) == length(expecteds), network != []} do
      {true, true} -> :ok
      {_true, false} -> {:error, "network hasn't been defined"}
      {false, _true} -> {:error, "test_input list and expecteds list are not equal"}
    end
  end

  # defp is_one_hot_encoded(encoded_list) do
  #   Enum.reduce_while(encoded_list, true,
  #   fn x, _acc ->
  #     case Matrex.apply(x, fn x -> if x >= 1, do: 1, else: 0 end) |> Matrex.sum() do
  #       1 -> {:cont, true}
  #       _ -> {:halt, false}
  #     end
  #   end
  #   )
  # end
end
