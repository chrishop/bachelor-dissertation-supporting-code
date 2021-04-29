defmodule RingTmp.ParameterServerTest do
  use ExUnit.Case

  alias RingTmp.ParameterServer
  alias FeedForwardNetwork.DefineNetwork

  describe "chunk_equally" do
    test "split equal" do

      nodes = 3
      training_list = [1,2,3,4,5]
      training_list_2 = [1,2,3,4,5,6]

      assert [[1,2,3]] == ParameterServer.chunk_equally(training_list, nodes)
      assert [[1,2,3], [4,5,6]] == ParameterServer.chunk_equally(training_list_2, nodes)
    end
  end

  describe "attach_supervisor" do
    test "each training pair gets its own supervisor" do

      nodes = 2
      node_list = [{RingTmp.TaskSupervisor, :bob@localhost}, {RingTmp.TaskSupervisor, :chris@localhost}]
      chunk_list = [
        [{:input1, :label1}, {:input2, :label2}],
        [{:input3, :label3}, {:input4, :label4}]
      ]

      expected = [
        [
          {{RingTmp.TaskSupervisor, :bob@localhost}, {:input1, :label1}},
          {{RingTmp.TaskSupervisor, :chris@localhost}, {:input2, :label2}}
        ],
        [
          {{RingTmp.TaskSupervisor, :bob@localhost}, {:input3, :label3}},
          {{RingTmp.TaskSupervisor, :chris@localhost}, {:input4, :label4}}
        ]
      ]

      assert expected == ParameterServer.attach_supervisor(chunk_list, node_list)
    end
  end

  describe "send_to_worker" do
    test "sends to worker" do

      {:ok, network} = DefineNetwork.define_network([{:output_layer, 4, 3}])

      input = Matrex.new(
        [
          [0.1], [0.2], [0.3], [0.4]
        ])

      label = Matrex.new([
        [0], [1], [0]
      ])

      train_element = {RingTmp.TaskSupervisor, {input, label}}

      assert [{:output_layer, _weight, _bias}] = ParameterServer.send_to_worker(train_element, network) |> Task.await()
    end
  end

  describe "send_to_workers" do
    test "sends to workers" do

      {:ok, network} = DefineNetwork.define_network([{:output_layer, 4, 3}])

      input_1 = Matrex.new([
          [0.1], [0.2], [0.3], [0.4]
      ])

      input_2 = Matrex.new([
          [0.2], [0.3], [0.4], [0.5]
      ])

      label_1 = Matrex.new([
        [0], [1], [0]
      ])

      label_2 = Matrex.new([
        [0], [0], [1]
      ])

      train_elements = [
        {RingTmp.TaskSupervisor, {input_1, label_1}},
        {RingTmp.TaskSupervisor, {input_2, label_2}}
      ]

      assert [[{:output_layer, _weight1, _bias1}], [{:output_layer, _weight2, _bias2}]] = ParameterServer.send_to_workers(train_elements, network)
    end
  end


  describe "avg_networks" do
    test "average networks" do
       # tested that this works correctly
      w_a = Matrex.new([[2, 4],[50, 2]])
      w_b = Matrex.new([[10, 10],[10, 10]])
      w_c = Matrex.new([[6, 4],[100, 4]])
      w_d = Matrex.new([[20, 20],[20, 20]])

      b_a = Matrex.new([[1],[2]])
      b_b = Matrex.new([[100],[10]])
      b_c = Matrex.new([[3],[4]])
      b_d = Matrex.new([[200],[20]])

      net_1 = [{:hidden_layer, w_a, b_a}, {:output_layer, w_b, b_b}]
      net_2 = [{:hidden_layer, w_c, b_c}, {:output_layer, w_d, b_d}]

      wr_1 = Matrex.new([[4, 4], [75, 3]])
      wr_2 = Matrex.new([[15, 15], [15, 15]])

      br_1 = Matrex.new([[2], [3]])
      br_2 = Matrex.new([[150], [15]])

      net_r = [{:hidden_layer, wr_1, br_1}, {:output_layer, wr_2, br_2}]

      assert net_r == ParameterServer.avg_networks([net_1, net_2])
    end
  end

  describe "single_loop" do
    test "a single loop" do
      {:ok, network} = DefineNetwork.define_network([{:output_layer, 4, 3}])

      input_1 = Matrex.new([
          [0.1], [0.2], [0.3], [0.4]
      ])

      input_2 = Matrex.new([
          [0.2], [0.3], [0.4], [0.5]
      ])

      label_1 = Matrex.new([
        [0], [1], [0]
      ])

      label_2 = Matrex.new([
        [0], [0], [1]
      ])

      chunk = [
        {RingTmp.TaskSupervisor, {input_1, label_1}},
        {RingTmp.TaskSupervisor, {input_2, label_2}}
      ]

      assert [{:output_layer, _weights, _bias}] = ParameterServer.single_loop(chunk, network, [learning_rate: 0.001])

    end
  end
end
