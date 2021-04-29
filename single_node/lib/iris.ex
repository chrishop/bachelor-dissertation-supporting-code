defmodule Iris do
  NimbleCSV.define(IrisParser, separator: ",", escape: "\"")

  def train_test_data(test_train_split \\ 1) do
    all = Enum.zip(features(), one_hot_encoded())
    |> Enum.shuffle()

    breakpoint = round(test_train_split * length(all))
    train_set = all |> Enum.slice(0, breakpoint)
    test_set = all |> Enum.slice(breakpoint, length(all) - breakpoint)

    {train_set, test_set}
  end

  def features() do
    iris_list()
    |> create_input_vector_list()
  end

  @spec one_hot_encoded :: list
  def one_hot_encoded() do
    all_vals = iris_list()
    |> Enum.map(fn x -> Enum.at(x, 4) end)

    unique_vals = all_vals
    |> Enum.uniq()

    {_index, mapping} = unique_vals
    |> Enum.with_index(1)
    |> Enum.reduce(
      {length(unique_vals), %{}},
      fn {val, index}, {len, acc} ->
        n_map = Map.put(acc, val, single_hot_encode(index, len))
        {len, n_map}
      end
    )

    all_vals
    |> Enum.map(fn x -> mapping[x] end)
  end

  # input setosa, versicolor, virginica to get binary expectations of each
  # flower against the others
  def get_binary_expecteds(flower) do
    "datasets/iris.csv"
    |> File.read!()
    |> IrisParser.parse_string()
    |> create_expected_list(flower)
  end

  defp iris_list() do
    "datasets/iris.csv"
    |> File.read!()
    |> IrisParser.parse_string()
  end

  # creates a list of 1x4 vectors
  defp create_input_vector_list(a_list) do
    a_list
    |> Enum.map(fn x ->
      x
      |> Enum.take(4)
      |> Enum.map(&String.to_float/1)
      |> (fn x -> Enum.map(x, fn y -> [y] end) end).()
      |> Matrex.new()
    end)
  end

  # creates one big 150x4 matrix
  # defp create_input_vector_list(a_list) do
  #   a_list
  #   |> Enum.map(fn x ->
  #     x
  #     |> Enum.take(4)
  #     |> IO.inspect()
  #     |> Enum.map(&String.to_float/1)
  #   end)
  #   |> Matrex.new()
  # end

  defp single_hot_encode(pos, length) do
    Matrex.zeros(length, 1)
    |> Matrex.update(pos, 1, fn _ -> 1 end)
  end

  defp create_expected_list(a_list, flower) do
    a_list
    |> Enum.map(&List.last/1)
    |> flower_list_to_int(flower)
    |> Matrex.new()
  end

  defp flower_list_to_int(flower_type_list, flower) do
    Enum.map(flower_type_list, fn x -> if x == flower, do: [1.0], else: [0.0] end)
  end

  # defp insert_list_row_to_matrix(a_list) do
  #   Matrex.new([a_list])
  # end
end
