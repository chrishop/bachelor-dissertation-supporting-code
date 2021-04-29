defmodule RingTmp.NeuralNet.HiddenLayer do
  def forward(prev_activation, weights, bias, _opts \\ []) do
    # TODO do I need to transpose weights?
    # add test with different size matrcies
    z_vec =
      Matrex.dot_tn(weights, prev_activation)
      |> Matrex.add(bias)

    activation = relu(z_vec)

    {z_vec, activation}
  end

  def back(z_vec, prev_activation, weights, bias, remaining_error, opts \\ []) do
    learning_rate = Keyword.get(opts, :learning_rate, 1.0)
    a_wrt_z = Matrex.apply(z_vec, fn value, _index -> if value > 0, do: 1, else: 0 end)
    cost_wrt_z = Matrex.multiply(remaining_error, a_wrt_z)

    # IO.inspect("z_vec")
    # IO.inspect(z_vec)

    # IO.inspect("a_wrt_z")
    # IO.inspect(a_wrt_z)

    # IO.inspect("cost_wrt_z")
    # IO.inspect(cost_wrt_z)

    # IO.inspect("recieved_error")
    # IO.inspect(remaining_error)

    # weight_delta = Matrex.multiply(a_wrt_z, remaining_error)
    # |> Matrex.dot_nt(Matrex.multiply(prev_activation[:rows], prev_activation))
    # |> IO.inspect
    # |> Matrex.multiply(learning_rate)

    weight_delta =
      prev_activation
      |> Matrex.dot_nt(cost_wrt_z)
      |> Matrex.multiply(learning_rate)

    bias_delta = cost_wrt_z |> Matrex.multiply(learning_rate)

    remaining_error = Matrex.dot(weights, cost_wrt_z)
    new_weights = weights |> Matrex.add(weight_delta, 1, 1)
    new_bias = Matrex.add(bias, bias_delta)

    # IO.inspect("weight delta layer")
    # IO.inspect(weight_delta)

    # IO.inspect("new_weights")
    # IO.inspect(new_weights)
    # IO.inspect("new_bias")
    # IO.inspect(new_bias)
    # IO.inspect("remaining_error")
    # IO.inspect(remaining_error)
    # IO.puts("")

    {new_weights, new_bias, remaining_error}
  end

  defp relu(z_vec) do
    z_vec |> Matrex.apply(fn val, _index -> if val > 0, do: val, else: 0 end)
  end

  # defp a_wrt_z(z_vec) do
  #   z_vec |> Matrex.apply(fn value, _index -> if value > 0, do: 1, else: 0 end)
  # end

  # defp z_wrt_w(prev_activations) do
  #   Matrex.dot(prev_activations, Matrex.ones(1,3))
  # end

  # defp weight_delta(prev_activations, remaining_error, z_wrt_w, a_wrt_z, learning_rate) do
  #   Matrex.dot_nt(z_wrt_w, remaining_error)
  #   |> Matrex.dot(z_wrt_w(prev_activations))
  #   |> Matrex.multiply(learning_rate)
  #   # later put this at the start so it only needs to do it 3 times
  # end

  # defp bias_delta(remaining_error, a_wrt_z, learning_rate) do
  #   remaining_error |> Matrex.multiply(z_wrt_w) |> Matrex.multiply(learning_rate)
  # end

  # defp layer_error(remaining_error, z_wrt_w, weights) do
  #   bias_delta = bias_delta(remaining_error, z_wrt_w, 1)
  #   weights |> Matrex.multiply(bias_delta)
  # end
end
