defmodule Mix.Tasks.RunMnist do
  use Mix.Task

  alias RingTmp.AppNode
  alias RingTmp.NeuralNet
  alias RingTmp.Datasets.Mnist

  @impl Mix.Task
  def run([config_name]) do

    Mix.Task.run("app.start")

    config_name = String.to_atom(config_name)

    # IO.inspect(Application.get_env(:ring_tmp, config_name))
    node_type = Application.get_env(:ring_tmp, config_name)[:app_node][:node_type]
    master_n = Application.get_env(:ring_tmp, config_name)[:app_node][:master_node] |> make_dest()
    this_n = Application.get_env(:ring_tmp, config_name)[:app_node][:this_node] |> make_dest()
    prev_n = Application.get_env(:ring_tmp, config_name)[:app_node][:prev_node] |> make_dest()
    next_n = Application.get_env(:ring_tmp, config_name)[:app_node][:next_node] |> make_dest()


    neural_net_definition = Application.get_env(:ring_tmp, config_name)[:neural_net][:definition]
    # IO.inspect(neural_net_definition)
    neural_net_opts = Application.get_env(:ring_tmp, config_name)[:neural_net][:opts]
    {:ok, _} = NeuralNet.define_network(neural_net_definition)



    DynamicSupervisor.start_child(
      RingTmp.NodeSupervisor,
      {AppNode, {node_type, master_n, this_n, prev_n, next_n, neural_net_opts}}
    )

    if (node_type == :master) do
      IO.puts("loading training data")
      {train_labels, train_inputs} = Mnist.train_data()

      IO.puts("loading test data")
      {test_labels, test_inputs} = Mnist.test_data()

      # IO.puts("loading validation data")
      # {validation_labels, validation_inputs} = Mnist.validation_data()

      IO.puts("wrapping bids")
      train_inputs = train_inputs |> wrap_with_bid()
      train_labels = train_labels |> wrap_with_bid()

      test_inputs = test_inputs |> wrap_with_bid()
      test_labels = test_labels |> wrap_with_bid()

      # validation_inputs = validation_inputs |> wrap_with_bid()
      # validation_labels = validation_labels |> wrap_with_bid()

      # train_inputs = Enum.take(train_inputs, 5) #|> IO.inspect()
      # train_labels = Enum.take(train_labels, 5) #|> IO.inspect()

      # IO.puts("creating train sample data")
      # train_input_sample = Enum.take(train_inputs, 1000)
      # train_labels_sample = Enum.take(train_labels, 1000)



      # IO.puts("press enter when all nodes set:")
      # IO.read(:stdio, :line)


      IO.puts("epoch_no:0")
      IO.puts("epoch_time:0")
      IO.puts("test data")
      AppNode.test(test_inputs, test_labels)
      await()
      Process.sleep(1000)
      IO.puts("train_data")
      AppNode.test(train_inputs, train_labels)
      await()
      Process.sleep(1000)

      epochs = 500
      Enum.each(1..epochs, fn ep_no ->
        IO.puts("epoch_no: #{ep_no}")
        t1 = :os.system_time(:millisecond)
        AppNode.train(train_inputs, train_labels)
        await()
        t2 = :os.system_time(:millisecond)
        ep_time = t2 - t1
        IO.puts("epoch_time:#{ep_time}")
        IO.puts("test data")
        AppNode.test(test_inputs, test_labels)
        await()
        Process.sleep(1000)
        IO.puts("train_data")
        AppNode.test(train_inputs, train_labels)
        await()
        Process.sleep(1000)
        # IO.puts("test on train data")
        # AppNode.test(train_inputs, train_labels)
        # await()
      end)

      IO.read(:stdio, :line)
    else
      IO.puts("press enter when finished with worker node:")
      IO.read(:stdio, :line)
    end
  end

  defp make_dest(hostname) do
    {RingTmp.AppNode, hostname}
  end

  defp await() do
    if AppNode.busy? do
      Process.sleep(100)
      await()
    end
  end

  defp wrap_with_bid(a_list) do
    a_list
    |> Enum.with_index()
    |> Enum.map(fn {elem, i} -> {i, elem} end)
  end

end
