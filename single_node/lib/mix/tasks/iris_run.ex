defmodule Mix.Tasks.IrisRun do
  use Mix.Task

  @shortdoc "trains then tests neural network on Iris dataset"
  def run(_) do
    # get training and testing data
    {train_set, test_set} = Iris.train_test_data(0.8)

    {train_inputs, train_examples} = Enum.unzip(train_set)
    {test_inputs, test_examples} = Enum.unzip(test_set)

    # start link with neural network
    FeedForwardNetwork.start_link([])

    IO.puts("epoch,train_accuracy,train_cost,test_accuracy,test_cost")
    {:ok, _network} = FeedForwardNetwork.define_network([{:hidden_layer, 4 , 4}, {:output_layer, 4, 3}])
    # {:ok, network} = FeedForwardNetwork.define_network([{:output_layer, 4, 3}])
    {:ok, test_accuracy, test_cost} = FeedForwardNetwork.test(test_inputs, test_examples)
    {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(train_inputs, train_examples)

    IO.puts("0,#{train_accuracy},#{train_cost},#{test_accuracy},#{test_cost}")

    epochs = 50
    Enum.each(
      1..epochs,
      fn ep_no ->
        epoch(ep_no,{train_inputs, train_examples, test_inputs, test_examples})
    end)
  end

  defp epoch(ep_no, {train_inputs, train_examples, test_inputs, test_examples}) do
    {:ok, _new_network} = FeedForwardNetwork.train(train_inputs, train_examples, learning_rate: 0.01)

    {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(train_inputs, train_examples)
    {:ok, test_accuracy, test_cost} = FeedForwardNetwork.test(test_inputs, test_examples)

    IO.puts("#{ep_no},#{train_accuracy},#{train_cost},#{test_accuracy},#{test_cost}")
    {train_inputs, train_examples, test_inputs, test_examples}
  end
end
