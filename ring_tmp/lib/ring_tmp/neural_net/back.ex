defmodule RingTmp.NeuralNet.Back do
  alias RingTmp.NeuralNet.Forward
  alias RingTmp.NeuralNet.CategoricalOutputLayer
  alias RingTmp.NeuralNet.HiddenLayer

  def back(train_inputs, expecteds, network, opts \\ []) do
    {new_network, _opts} =
      Enum.zip(train_inputs, expecteds)
      |> Enum.reduce({network, opts}, fn {input, expected}, {network, opts} ->
        {remaining_error, new_network} = back_once(input, expected, network, opts)
        {new_network, opts}
      end)

    new_network
  end

  # def back_node(a_in, remaining_error, network, opts \\ []) do
  #   output_tuples = Forward.forward_list(network, a_in)
  #   [first | offset] = output_tuples
  #   cur_tuples = Enum.drop(output_tuples, -1)
  #   prev_tuples = offset

  #   {remaining_error, _opts, new_network} =
  #     Enum.zip([cur_tuples, prev_tuples, Enum.reverse(network)])
  #     |> Enum.reduce({remaining_error, opts, []}, fn x, acc ->
  #       back_reduce(x, acc)
  #     end)

  #   # # need to also return remaining error here
  #   # {remaining_error, new_network}
  # end

  # you can also read expecteds as remaining error
  def back_once(train_input, expected, network, opts \\ []) do
    output_tuples = Forward.forward_list(network, train_input)
    [first | offset] = output_tuples
    cur_tuples = Enum.drop(output_tuples, -1)
    prev_tuples = offset

    # is this a better way to offset?
    # [activations, offset] = activations_list |> Enum.chunk_every(length(activations_list)-1, 2)

    # the accumulator stores the remaining error of the previous layer
    # and the changes to the weights and bias for each layer

    {remaining_error, _opts, new_network} =
      Enum.zip([cur_tuples, prev_tuples, Enum.reverse(network)])
      |> Enum.reduce({expected, opts, []}, fn x, acc ->
        back_reduce(x, acc)
      end)

    # need to also return remaining error here
    {remaining_error, new_network}
  end

  defp back_reduce(
         {{z, a}, {_prev_z, prev_a}, {layer_type, _, _} = layer},
         {remaining_error, opts, acc}
       ) do
    {n_weights, n_bias, remaining_error} = back_layer(layer, a, z, prev_a, remaining_error, opts)
    {remaining_error, opts, [{layer_type, n_weights, n_bias} | acc]}
  end

  defp back_layer(
         {:output_layer, weights, bias},
         activation,
         _z_vec,
         prev_activation,
         target_activation,
         opts
       ) do
    CategoricalOutputLayer.back(
      activation,
      prev_activation,
      weights,
      bias,
      target_activation,
      opts
    )
  end

  defp back_layer(
         {:hidden_layer, weights, bias},
         _activation,
         z_vec,
         prev_activation,
         remaining_error,
         opts
       ) do
    HiddenLayer.back(z_vec, prev_activation, weights, bias, remaining_error, opts)
  end
end
