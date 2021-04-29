defmodule RingTmp.Datasets.Mnist do
  NimbleCSV.define(MnistParser, separator: ",", escape: "\"")

  def train_data(size \\ 100000) do
    {labels, input_list} = mnist_train_list() |> Enum.take(size)
    |> split_label_from_data()

    matrix_list = input_list |> row_to_matrix()

    encoded_labels = hot_encode(labels, 10)

    {encoded_labels, matrix_list}
  end

  def test_data(size \\ 100000) do
    {labels, input_list} = mnist_test_list() |> Enum.take(size)
    |> split_label_from_data()

    matrix_list = input_list |> row_to_matrix()

    encoded_labels = hot_encode(labels, 10)

    {encoded_labels, matrix_list}
  end

  def validation_data() do
    test_data(500)
  end



  defp mnist_train_list() do
    "datasets/mnist_train.csv"
    |> File.read!()
    |> MnistParser.parse_string()
  end

  defp mnist_test_list() do
    "datasets/mnist_test.csv"
    |> File.read!()
    |> MnistParser.parse_string()
  end

  defp split_label_from_data(label_and_data) do
    label_and_data
    |> Enum.map(fn [label| data] -> {label, data} end)
    |> Enum.unzip()
  end

  defp row_to_matrix(data_list) do
    data_list
    |> Enum.map(
      fn row ->
        row
        |> Enum.map(&String.to_integer/1)
        |> Enum.map(fn elem -> elem / 255 end)
        |> Enum.map(fn elem -> [elem] end)
        |> Matrex.new()
      end
    )
  end


  defp hot_encode(labels, categories) do
    labels
    |> Enum.map(
      fn label ->
        label
        |> String.to_integer()
        |> single_hot_encode(categories)
    end)
  end

  defp single_hot_encode(pos, length) do
    Matrex.zeros(length, 1)
    |> Matrex.update(pos + 1, 1, fn _ -> 1 end)
  end
end
