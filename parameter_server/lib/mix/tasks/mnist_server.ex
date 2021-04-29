defmodule Mix.Tasks.MnistServer do
  use Mix.Task

  alias RingTmp.TaskSupervisor
  alias FeedForwardNetwork.DefineNetwork
  alias FeedForwardNetwork.Back
  alias FeedForwardNetwork.Test
  alias RingTmp.ParameterServer




  def run(_) do
    Mix.Task.run("app.start")
    Node.start(:alice@localhost, :shortnames)

    # get training and testing data

    IO.puts("loading train data")
    {train_labels, train_inputs} = Mnist.train_data()

    IO.puts("loading test data")
    {test_labels, test_inputs} = Mnist.test_data()

    IO.puts("zipping")
    train_set = Enum.zip(train_inputs, train_labels)
    test_set = Enum.zip(test_inputs, test_labels)

    # List.first(train_set) |> IO.inspect()

    # define network in parameter server
    {:ok, network} = DefineNetwork.define_network([{:hidden_layer, 784 , 50}, {:output_layer, 50, 10}])

    sups = [
      {RingTmp.TaskSupervisor, :b@localhost},
      {RingTmp.TaskSupervisor, :c@localhost},
      {RingTmp.TaskSupervisor, :d@localhost},
      {RingTmp.TaskSupervisor, :e@localhost},
      {RingTmp.TaskSupervisor, :f@localhost},
      {RingTmp.TaskSupervisor, :g@localhost}
    ]

    {:ok, start_train_acc, start_train_cost} = Test.test(train_inputs, train_labels, network)
    {:ok, start_test_acc, start_test_cost} = Test.test(test_inputs, test_labels, network)

    # train in parallel
    IO.puts("")
    IO.puts("epoch,train_accuracy,train_cost,test_accuracy,test_cost,ep_time")
    IO.puts("#{0},#{start_train_acc},#{start_train_cost},#{start_test_acc},#{start_test_cost},0")

    epoch(20, train_set, test_set, sups, network, learning_rate: 0.01)



  end

  defp epoch(epochs, train_set, test_set, sups, network, opts) do

    Enum.reduce(1..epochs, network, fn ep_no, c_network ->
      t1 = :os.system_time(:millisecond)
      new_network = ParameterServer.run(train_set, sups, c_network, opts)
      t2 = :os.system_time(:millisecond)
      total_time = t2 - t1

      {train_inputs, train_labels} = Enum.unzip(train_set)
      {test_inputs, test_labels} = Enum.unzip(test_set)

      {:ok, train_acc, train_cost} = Test.test(train_inputs, train_labels, new_network)
      {:ok, test_acc, test_cost} = Test.test(test_inputs, test_labels, new_network)

      IO.puts("#{ep_no},#{train_acc},#{train_cost},#{test_acc},#{test_cost},#{total_time}")
      new_network
    end)
  end
end
