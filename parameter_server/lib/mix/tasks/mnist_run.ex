defmodule Mix.Tasks.MnistRun do
  use Mix.Task

  @impl Mix.Task
  def run(_) do

    # number of records

    IO.puts("loading train data")
    {train_labels, train_inputs} = Mnist.train_data()

    IO.puts("loading test data")
    {test_labels, test_inputs} = Mnist.test_data()

    {validation_labels, validation_inputs} = Mnist.validation_data()

    FeedForwardNetwork.start_link([])

    {:ok, _network} = FeedForwardNetwork.define_network([{:hidden_layer, 784 , 50}, {:output_layer, 50, 10}])

    IO.puts("calculating initial acc/cost")
    {:ok, test_accuracy, test_cost} = FeedForwardNetwork.test(test_inputs, test_labels)
    {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(train_inputs, train_labels)

    IO.puts("")
    IO.puts("start")
    IO.puts("train accuracy/cost: #{train_accuracy}/#{train_cost}")
    IO.puts("test accuracy/cost: #{test_accuracy}/#{test_cost}")
    IO.puts("")

    epochs = 20

    IO.puts("epoch,train_accuracy,train_cost,validation_accuracy,validation_cost,epoch_time")
    Enum.each(
      1..epochs,
      fn ep_no ->
        epoch(ep_no,{train_inputs, train_labels, test_inputs, test_labels})
    end)

    {:ok, test_accuracy, test_cost} = FeedForwardNetwork.test(test_inputs, test_labels)
    {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(train_inputs, train_labels)

    IO.puts("")
    IO.puts("final acc_cost")
    IO.puts("train accuracy/cost: #{train_accuracy}/#{train_cost}")
    IO.puts("test accuracy/cost: #{test_accuracy}/#{test_cost}")
  end

  defp epoch(ep_no, {train_inputs, train_labels, validation_inputs, validation_labels}) do
    t1 = :os.system_time(:millisecond)
    {:ok, _new_network} = FeedForwardNetwork.train(train_inputs, train_labels, learning_rate: 0.01)
    t2 = :os.system_time(:millisecond)
    # train_input_sample = Enum.take(train_inputs, 1000)
    # train_labels_sample = Enum.take(train_labels, 1000)

    {:ok, train_accuracy, train_cost} = FeedForwardNetwork.test(train_inputs, train_labels)
    {:ok, test_accuracy, test_cost} = FeedForwardNetwork.test(validation_inputs, validation_labels)

    ep_time = t2-t1
    IO.puts("#{ep_no},#{train_accuracy},#{train_cost},#{test_accuracy},#{test_cost},#{ep_time}")
    {train_inputs, train_labels, validation_inputs, validation_labels}
  end
end
