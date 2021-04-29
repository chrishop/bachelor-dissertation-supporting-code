defmodule Mix.Tasks.RunNode do
  use Mix.Task

  alias RingTmp.AppNode
  alias RingTmp.NeuralNet

  @impl Mix.Task
  def run([config_name]) do

    Mix.Task.run("app.start")

    config_name = String.to_atom(config_name)

    node_type = Application.get_env(:ring_tmp, config_name)[:app_node][:node_type]
    master_n = Application.get_env(:ring_tmp, config_name)[:app_node][:master_node] |> make_dest()
    this_n = Application.get_env(:ring_tmp, config_name)[:app_node][:this_node] |> make_dest()
    prev_n = Application.get_env(:ring_tmp, config_name)[:app_node][:prev_node] |> make_dest()
    next_n = Application.get_env(:ring_tmp, config_name)[:app_node][:next_node] |> make_dest()

    neural_net_definition = Application.get_env(:ring_tmp, config_name)[:neural_net][:definition]

    {:ok, _} = NeuralNet.define_network(neural_net_definition, 42)



    DynamicSupervisor.start_child(
      RingTmp.NodeSupervisor,
      {AppNode, {node_type, master_n, this_n, prev_n, next_n}}
    )



    if (node_type == :master) do
      IO.puts("press enter when all nodes set:")
      IO.read(:stdio, :line)

      batches = [
        {
          0,
          Matrex.new([
            [1],
            [2],
            [3],
            [4]
          ])
        },
        {
          1,
          Matrex.new([
            [4],
            [5],
            [6],
            [7]
          ])
        }
      ]

      expecteds = [
        {
          0,
          Matrex.new([
            [0],
            [1],
            [0]
          ])
        },
        {
          1,
          Matrex.new([
            [0],
            [0],
            [1]
          ])
        }
      ]



      AppNode.train(batches, expecteds)
      await()
      IO.puts("")
      AppNode.test(batches, expecteds)

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

end
