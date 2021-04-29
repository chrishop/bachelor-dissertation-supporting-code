defmodule RingTmp.ParameterServer do

  alias FeedForwardNetwork.Back

  def run(training_pairs, node_list, network, opts) do
    node_no = Enum.count(node_list)

    chunk_list = training_pairs
    |> chunk_equally(node_no)
    |> attach_supervisor(node_list)

    multi_loop(chunk_list, network, opts)
  end

  def multi_loop(chunk_list, network, opts) do
    chunk_list
    |> Enum.reduce(
      network,
      fn chunk, prev_network ->
        single_loop(chunk, prev_network, opts)
      end
    )
  end

  def single_loop(chunk, network, opts) do
    chunk
    |> send_to_workers(network, opts)
    |> avg_networks()
  end

  # returns list of networks
  def send_to_workers(chunk, network, opts \\ []) do
    chunk
    |> Enum.map(
      fn sup_train_pair ->
        send_to_worker(sup_train_pair, network, opts)
      end)
    |> Enum.map(fn t -> Task.await(t) end)
  end


  def send_to_worker({sup, {train_input, train_label}}, network, opts \\ []) do
    # IO.inspect({"sup", sup})
    # IO.inspect({"train_input", train_input})
    # IO.inspect({"train_label", train_label})
    # IO.inspect({"network", network})
    # IO.inspect({"opts", opts})

    # sup
    # |> Task.Supervisor.async(
    #   FeedForwardNetwork.Back,
    #   :back_once,
    #   [train_input, train_label, network, opts]
    # )

    Task.async(
      FeedForwardNetwork.Back,
      :back_once,
      [train_input, train_label, network, opts]
    )
  end

  def avg_networks(networks) do
    size = Enum.count(networks)

    networks
    |> Enum.reduce(
      fn network, sum_acc ->
        sum_network(network, sum_acc)
      end)
    |> divide_network(size)
  end

  def attach_supervisor(training_chunks, sup_list) do
    training_chunks
    |> Enum.map(
      fn chunk ->
        Enum.zip(sup_list, chunk)
      end
    )
  end

  def chunk_equally(training_pairs, node_no) do
    Enum.chunk_every(training_pairs, node_no, node_no, :discard)
  end



  defp sum_network(network_a, network_b) do
    Enum.zip(network_a, network_b)
    |> Enum.map(
      fn {layer_a, layer_b} ->
        sum_layer(layer_a, layer_b)
      end
    )
  end

  defp sum_layer(layer_a, layer_b) do
    {layer_t_a, weights_a, bias_a} = layer_a
    {layer_t_b, weights_b, bias_b} = layer_b

    weights_sum = Matrex.add(weights_a, weights_b)
    bias_sum = Matrex.add(bias_a, bias_b)

    if layer_t_a == layer_t_b do
      {layer_t_a, weights_sum, bias_sum}
    else
      {:error, "not same layer_type"}
    end
  end

  defp divide_network(a_network, divisor) do
    a_network
    |> Enum.map(
      fn layer ->
        divide_layer(layer, divisor)
      end
    )
  end

  defp divide_layer({a_layer, a_weight, a_bias}, divisor) do
    div_weight = Matrex.divide(a_weight, divisor)
    div_bias = Matrex.divide(a_bias, divisor)

    {a_layer, div_weight, div_bias}
  end
end
