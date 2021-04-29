defmodule FeedForwardNetwork.Predict do
  alias FeedForwardNetwork.Forward

  def prediction(input_vector, network) do

    Forward.forward(network, input_vector)
    |> predict_output()
    |> (fn x -> {:ok, x} end).()

    # case validate_prediction_input(input_vector, network) do
    #   :ok ->



    #   {:error, msg} ->
    #     {:error, msg}
    # end
  end

  defp predict_output(output_vector) do
    # alternative way could be to map
    # fn x -> if x > 1/output_vector_length
    # may be faster, less cumbersome

    {row, col} = Matrex.size(output_vector)
    zeros = Matrex.zeros(row, col)
    make_prediction = fn row, zeros -> Matrex.update(zeros, row, 1, fn _x -> 1 end) end

    output_vector
    |> Matrex.argmax()
    |> make_prediction.(zeros)
  end

  # TODO validate prediction input
  # defp validate_prediction_input(input, network) do
  #   case {Matrex.size(input), List.first(network)} do
  #     {{x, _}, {_layer, y, _, _}} when x == y ->
  #       :ok

  #     {{x, _}, {_layer, y, _, _}} ->
  #       {:error, "wrong input length for network, expected #{y} given #{x}"}

    #   {_, nil} ->
    #     {:error, "network hasn't been defined yet"}

    #   _ ->
    #     {:error, "something very wrong with input or network state"}
    # end
  # end
end
