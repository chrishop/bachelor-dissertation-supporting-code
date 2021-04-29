defmodule Mix.Tasks.IrisServer do
  use Mix.Task

  alias RingTmp.TaskSupervisor
  alias FeedForwardNetwork.DefineNetwork
  alias FeedForwardNetwork.Back
  alias FeedForwardNetwork.Test
  alias RingTmp.ParameterServer




  def run(_) do
    Mix.Task.run("app.start")
    Node.start(:alice@localhost, :shortnames)
    Node.connect(:bob@localhost)

    # get training and testing data
    {train_set, test_set} = Iris.train_test_data(0.8)

    {train_inputs, train_labels} = Enum.unzip(train_set)
    {test_inputs, test_labels} = Enum.unzip(test_set)

    # worker_nodes = 2

    # define network in parameter server
    {:ok, network} = DefineNetwork.define_network([{:output_layer, 4, 8}, {:hidden_layer, 8, 8}, {:output_layer, 8, 3}])

    sups = [
      {RingTmp.TaskSupervisor, :bob@localhost}, {RingTmp.TaskSupervisor, :chris@localhost}
    ]

    # test network before
    {:ok, start_train_acc, start_train_cost} = Test.test(train_inputs, train_labels, network)
    {:ok, start_test_acc, start_test_cost} = Test.test(test_inputs, test_labels, network)

    # train in parallel
    IO.puts("epoch,train_accuracy,train_cost,test_accuracy,test_cost,ep_time")
    IO.puts("#{0},#{start_train_acc},#{start_train_cost},#{start_test_acc},#{start_test_cost},0")

    epoch(20, train_set, test_set, sups, network, learning_rate: 0.01)



  end

  defp epoch(epochs, train_set, test_set, sups, network, opts) do

    Enum.reduce(1..epochs, network, fn ep_no, c_network ->
      t1 = :os.system_time(:millisecond)
      new_network = ParameterServer.run(train_set, sups, c_network, opts)
      t2 = :os.system_time(:millisecond)
      {train_inputs, train_labels} = Enum.unzip(train_set)
      {test_inputs, test_labels} = Enum.unzip(test_set)

      ep_time = t2 - t1

      {:ok, train_acc, train_cost} = Test.test(train_inputs, train_labels, new_network)
      {:ok, test_acc, test_cost} = Test.test(test_inputs, test_labels, new_network)

      IO.puts("#{ep_no},#{train_acc},#{train_cost},#{test_acc},#{test_cost},#{ep_time}")
      new_network
    end)
  end
end
