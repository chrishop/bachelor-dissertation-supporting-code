defmodule RingTmp.AppNode do
  use GenServer

  alias RingTmp.NeuralNet

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  # returns :ok | {:error, reason}
  def train(batches, expecteds) do
    GenServer.call(__MODULE__, {:train, batches, expecteds}, :infinity)
  end

  # returns {accuracy, loss}
  def test(batches, expecteds) do
    GenServer.call(__MODULE__, {:test, batches, expecteds}, :infinity)
  end

  def busy?() do
    GenServer.call(__MODULE__, :busy?, 20000)
  end


  @type node_type_t :: :master | :worker
  @type app_node_state :: {node_type_t, Process.dest(), Process.dest(), Process.dest(), Process.dest(), map(), map(), map(), non_neg_integer(), boolean(), any()}

  @type batch_t :: {number, any}


  @impl true
  def init({node_type, m_node, this_node, prev_node, next_node, nn_opts}) do

    this_host = elem(this_node, 1)
    Node.start(this_host, :shortnames)

    {:ok, {node_type, m_node, this_node, prev_node, next_node, %{}, %{}, %{accuracy: 0, cost: 0}, 0, false, nn_opts}}
  end


  @impl true
  def handle_call(
    {:train, all_batches, all_expecteds}, _from,
    state = {:master, _m_node, _this, _prev_node, _next_node, _history, _expecteds, _test_acc, _max_bid, true, _nn_opts}
  ) do
    {:reply, {:error, "busy"}, state}
  end

  @impl true
  def handle_call(
    {:train, all_batches, all_expecteds}, _from,
    state = {:master, _m_node, _this, prev_node, next_node, _history, _expecteds, test_acc, max_bid, false, _nn_opts}
  ) do

    # send_log(state, "*** TRAIN ***")

    # check all nodes up and running
    prev_host = elem(prev_node, 1)
    next_host = elem(next_node, 1)

    # start

    case Node.connect(prev_host) and Node.connect(next_host) do
      true ->
        # batch
        # send_log(state, "In start")

        Enum.map(all_batches, fn batch ->
          send(next_node, {:forward, forward_batch(batch)})
        end)

        # master needs to store the expected values of each
        # forward propagation so it can produce the remaining error

        expecteds = Enum.reduce(
          all_expecteds, %{},
          fn expected, acc -> add_history(acc, expected) end
        )

        history = Enum.reduce(
          all_batches, %{},
          fn batch, acc -> add_history(acc, batch) end
        )

        # get max bid and add it to the state
        max_bid  = elem(List.last(all_batches), 0)

        state = state
        |> put_elem(5, history)
        |> put_elem(6, expecteds)
        |> put_elem(8, max_bid)
        |> put_elem(9, true) # making master busy

        {:reply, :ok, state}

      _ ->
        {:reply, {:error, "#{prev_host} or #{next_host} not up"}, state}
    end
  end


  @impl true
  def handle_call(
    {:test, all_batches, all_expecteds}, _from,
    state = {:master, _m_node, _this, _prev_node, _next_node, _history, _expecteds, _test_acc, _max_bid, true, _nn_opts}
  ) do
    {:reply, {:error, "busy"}, state}
  end

  @impl true
  def handle_call(
    {:test, all_batches, all_expecteds}, _from,
    state = {:master, _m_node, _this, prev_node, next_node, history, expecteds, _test_acc, max_bid, false, _nn_opts}
  ) do

    # send_log(state, "*** TEST ***")

    # send all the batches
    Enum.map(all_batches, fn batch ->
      send(next_node, {:forward_test, forward_batch(batch)})
    end)

    # store expecteds in state
    expecteds = Enum.reduce(
      all_expecteds, %{},
      fn expected, acc -> add_history(acc, expected)
    end)

    # get max bid and add it to the state
    max_bid  = elem(List.last(all_batches), 0)

    state = state
    |> put_elem(6, expecteds)
    |> put_elem(7, %{accuracy: 0, cost: 0}) #reset testing accumulator
    |> put_elem(8, max_bid)
    |> put_elem(9, true) # making master busy

    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:busy?, _from, state) do
    {:reply, elem(state, 9), state}
  end


  ##### FORWARD PROP #####

  # first forward action action is called in here
  @impl true
  def handle_info({:forward, batch}, state = {:worker, _m_node, _this, _prev, next_node, history, _exp, _test_acc, _max_bid, busy, _nn_opts}) do
    # do forward action
    {num, data} = batch

    # IO.puts("In :forward :worker batch_id: #{num}")
    # send_log(state, "In :forward :worker batch_id: #{num}")

    send(next_node, {:forward, forward_batch(batch)})
    # {row, col} = Matrex.size(data)
    # batch_size = row * col
    # IO.puts(inspect(batch_size))

    history = add_history(history, batch)
    state = put_elem(state, 5, history)
    {:noreply, state}

  end

  @impl true
  # this is called from the last worker to loop around and hit the master node
  # now we are at the master node we can evaluate the results of the network
  def handle_info({:forward, batch}, state = {:master, _m_node, _this, prev_node, _next_node, _history, expecteds, _test_acc, _max_bid, busy, _nn_opts}) do

    # do back
    {b_id, b_data} = batch
    id_as_str = Integer.to_string(b_id)
    expected = Map.get(expecteds, id_as_str)




    # IO.puts("In :forward :master batch_id: #{num}")
    # send_log(state, "In :forward :master batch_id: #{b_id}")

    send(prev_node, {:back, {b_id, expected}})

    {:noreply, state}
  end


  ##### BACK PROP #####

  @impl true
  def handle_info({:back, error_batch}, state = {:worker, _m_node, _this, prev_node, _next, history, _exp, _test_acc, _max_bid, _busy, nn_opts}) do

    {b_id, error_vec} = error_batch
    b_id_str = b_id |> Integer.to_string()
    input_for_batch = Map.get(history, b_id_str)
    remaining_error = NeuralNet.back(input_for_batch, error_vec, nn_opts)



    # IO.puts("In :back :worker batch_id: #{num}")
    # send_log(state, "In :back :worker batch_id: #{b_id}")

    send(prev_node, {:back, {b_id, remaining_error}})

    # {row, col} = Matrex.size(remaining_error)
    # batch_size = row * col
    # IO.puts(inspect(batch_size))

    {:noreply, state}
  end

  # the data has made it back to the master node
  @impl true
  def handle_info({:back, error_batch}, state = {:master, _m_node, _this, _prev, _next, history, _exp, _test_acc, max_bid, _busy, nn_opts}) do

    {b_id, error_vec} = error_batch
    b_id_str = b_id |> Integer.to_string()
    input_for_batch = Map.get(history, b_id_str)
    remaining_error = NeuralNet.back(input_for_batch, error_vec, nn_opts)
    # send_log(state, "In :back :master batch_id: #{b_id}")

    if b_id == max_bid do
      # send_log(state, "Final bid #{b_id}")

      # making master not busy
      state = put_elem(state, 9, false)
      {:noreply, state}
    else
      {:noreply, state}
    end
  end


  ##### TEST #####


  @impl true
  def handle_info({:forward_test, {b_id, batch_data}}, state = {:master, _m_node, _this, _prev, _next, _history, expecteds, test_acc, max_bid, _busy, _nn_opts}) do

    # send_log(state, "In :forward_test :master batch_id: #{b_id}")

    accuracy_acc = Map.get(test_acc, :accuracy)
    cost_acc = Map.get(test_acc, :cost)

    b_id_str = b_id |> Integer.to_string()
    # the prediction is then checked
    expected = Map.get(expecteds, b_id_str)

    accuracy_acc = if predict_output(batch_data) == expected do
      accuracy_acc + 1
    else
      accuracy_acc
    end

    # the cost is summed to the ongoing cost
    loss = categorical_loss(batch_data, expected)
    cost_acc = cost_acc + loss

    state = put_elem(state, 7, %{accuracy: accuracy_acc, cost: cost_acc})

    # when the last packed has been recieved output the results
    if b_id == max_bid do
      average_acc = accuracy_acc / (max_bid + 1)
      average_cost = cost_acc / (max_bid + 1)
      #send_log(state, "Test Done,accuracy: #{average_acc},cost: #{average_cost}")
      send_log(state, "#{average_acc},#{average_cost}")

      # making master not busy
      state = put_elem(state, 9, false)
      {:noreply, state}
    else
      {:noreply, state}
    end



  end

  # if the worker doesn't contain a categorical layer do forward prop
  # if it does do a prediction and a
  @impl true
  def handle_info({:forward_test, batch}, state = {:worker, _m_node, _this, _prev, next_node, _history, _exp, _test_acc, _max_bid, _busy, _nn_opts}) do

    {b_id, _b_data} = batch

    # send_log(state, "In :forward_test :worker batch_id: #{b_id}")
    send(next_node, {:forward_test, forward_batch(batch)})

    {:noreply, state}
  end


  @impl true
  def handle_info({:log, msg}, state) do

    IO.inspect(msg)
    {:noreply, state}
  end


  @impl true
  def handle_info(msg, state) do
    IO.inspect("NO MATCH")
    IO.inspect(msg)

    {:noreply, state}
  end

  @spec send_log(app_node_state, String.t()) :: :ok
  defp send_log({_node_t, master_n, this_n, _p_n, _n_n, _hist, _exp, _test_acc, _max_bid, _busy, _nn_opts}, msg) do
    this_h = elem(this_n, 1)
    log_msg = "from: #{this_h}, msg: #{msg}"
    send(master_n, {:log, msg})
    :ok
  end

  defp forward_batch({num, batch_data}) do
    {num, NeuralNet.forward(batch_data)}
  end

  defp add_history(history, batch) do
    {b_id, b_data} = batch
    id_as_atom = b_id |> Integer.to_string()
    Map.put(history, id_as_atom, b_data)
  end

  defp categorical_loss(activation, target_activation) do
    target_position = get_target_category(target_activation)

    activation
    |> Matrex.at(target_position, 1)
    |> (fn x ->
      if x == 0.0 do
        100
      else
        -:math.log(x)
      end
    end).()
  end

  defp get_target_category(target_activation) do
    {row, _col} = Matrex.size(target_activation)

    Enum.map(1..row, fn x -> [x] end)
    |> Matrex.new()
    |> Matrex.multiply(target_activation)
    |> Matrex.sum()
    |> round()
  end

  defp predict_output(output_vector) do
    # alternative way could be to map
    # fn x -> if x > 1/output_vector_length
    # may be faster, less cumbersome

    {row, col} = Matrex.size(output_vector)
    zeros = Matrex.zeros(row, col)
    make_prediction = fn row, zeros -> Matrex.update(zeros, row, 1, fn _x -> 1 end) end

    output_vector
    |> Matrex.argmax()
    |> make_prediction.(zeros)
  end
end
