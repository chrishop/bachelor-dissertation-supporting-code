defmodule RingTmp.NeuralNet.DefineNetwork do
  def define_network(definition), do: init_network(definition, :rand.seed(:exsss))

  def define_network(definition, seed_val),
    do: init_network(definition, :rand.seed(:exsss, seed_val))

  defp init_network(definition, seed_state) do
    case validate_definitions(definition) do
      :ok ->
        {layers, _state} = initialise_parameters(definition, seed_state)
        {:ok, layers}

      {:error, message} ->
        {:error, message}
    end
  end

  # TODO Need to have an initial input
  # Not have the final output be the size of the input
  defp initialise_parameters(definition, initial_seed) do
    definition
    |> Enum.reduce(
      {[], initial_seed},
      fn layer, {acc, seed_state} ->
        {layer, new_state} = initialise_layer(layer, seed_state)
        {acc ++ [layer], new_state}
      end
    )
  end

  defp initialise_layer({layer_type, input_size, output_size}, seed_state) do
    init_type = layer_to_init(layer_type)

    {initial_weights, seed_state} =
      initialise_weights(input_size, output_size, init_type, seed_state)

    initial_bias = initialise_bias(output_size)

    {{layer_type, initial_weights, initial_bias}, seed_state}
  end

  # may have got col and rows mixed up
  defp initialise_weights(col, row, init_type, seed_state) do
    {_, _, _, seed_state, list_of_lists} =
      Enum.reduce(
        0..(col - 1),
        {row, col, init_type, seed_state, []},
        &random_row/2
      )

    {Matrex.new(list_of_lists), seed_state}
  end

  defp random_row(_x, {row, col, init_type, seed_state, acc}) do
    {_, _, new_state, row_vals} =
      Enum.reduce(
        0..(row - 1),
        {col, init_type, seed_state, []},
        &random_val/2
      )

    {row, col, init_type, new_state, [row_vals | acc]}
  end

  defp random_val(_x, {col, :he, seed_state, acc}) do
    {val, new_state} = :rand.normal_s(0, 2 / col, seed_state)
    {col, :he, new_state, [val | acc]}
  end

  defp random_val(_x, {col, :pos, seed_state, acc}) do
    {val, new_state} = :rand.normal_s(0.5, 0.25, seed_state)
    {col, :pos, new_state, [val | acc]}
  end

  defp random_val(_x, {col, :xavier, seed_state, acc}) do
    {val, new_state} = :rand.normal_s(0, 1 / col, seed_state)
    {col, :xavier, new_state, [val | acc]}
  end

  defp initialise_bias(col) do
    Matrex.zeros(col, 1)
  end

  defp layer_to_init(:hidden_layer), do: :pos
  defp layer_to_init(:output_layer), do: :xavier
  defp layer_to_init(layer_type), do: {:error, "unrecognised layer type #{layer_type}"}

  defp validate_definitions(definitions) do
    # TODO Chunk here to make sure output size of one layer
    # and input size of the next layer match
    case definitions do
      [] ->
        {:error, "empty definition"}

      x ->
        Enum.reduce(x, :ok, fn x, acc -> validate_definition(x, acc) end)
    end
  end

  defp validate_definition(definition, acc) do
    case acc do
      :ok ->
        case definition do
          {:hidden_layer, num_i, num_o} when num_i > 0 and num_o > 0 ->
            :ok

          {:output_layer, num_i, num_o} when num_i > 0 and num_o > 0 ->
            :ok

          a_def ->
            {:error, "#{inspect(a_def)} is not a valid definition"}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end
end
