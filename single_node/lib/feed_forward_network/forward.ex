defmodule FeedForwardNetwork.Forward do
  def forward(network, input_vector) do
    Enum.reduce(network, input_vector, fn x, acc -> forward_layer(x, acc) end)
  end

  # returns a list of activations of each layer in reverse order
  # e.g. [activation_n, activation_n-1, activation_n-n]
  # where n is layer number of the final layer
  def forward_list(network, input_vector) do
    Enum.reduce(network, [{:na, input_vector}], fn layer, acc ->
      [forward_layer_list(layer, List.first(acc)) | acc]
    end)
  end

  defp forward_layer_list(
         {:hidden_layer, weights, bias},
         {_prev_z_vec, prev_activation}
       ) do
    HiddenLayer.forward(prev_activation, weights, bias, [])
  end

  defp forward_layer_list(
         {:output_layer, weights, bias},
         {_prev_z_vec, prev_activation}
       ) do
    activation = CategoricalOutputLayer.forward(prev_activation, weights, bias, [])
    {:na, activation}
  end

  defp forward_layer({:hidden_layer, weights, bias}, prev_activation) do
    {_z_vec, activation} = HiddenLayer.forward(prev_activation, weights, bias, [])
    activation
  end

  defp forward_layer({:output_layer, weights, bias}, prev_activation) do
    CategoricalOutputLayer.forward(prev_activation, weights, bias, [])
  end
end
