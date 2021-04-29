defmodule FeedForwardNetwork do
  use GenServer

  alias FeedForwardNetwork.DefineNetwork
  alias FeedForwardNetwork.Predict
  alias FeedForwardNetwork.Test
  alias FeedForwardNetwork.Train

  # client
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def define_network(definition, seed_value) do
    GenServer.call(__MODULE__, {:define_network, definition, seed_value})
  end

  def define_network(definition) do
    GenServer.call(__MODULE__, {:define_network, definition})
  end

  def train(inputs, expecteds, opts \\ []) do
    GenServer.call(__MODULE__, {:train, inputs, expecteds, opts}, :infinity)
  end

  def test(inputs, expecteds) do
    GenServer.call(__MODULE__, {:test, inputs, expecteds}, :infinity)
  end

  def predict(input_vector) do
    GenServer.call(__MODULE__, {:predict, input_vector})
  end

  # callbacks

  @impl true
  def init(_opts) do
    {:ok, []}
  end

  @impl true
  def handle_call({:define_network, definition, seed_val}, _from, state) do
    case DefineNetwork.define_network(definition, seed_val) do
      {:ok, network} ->
        {:reply, {:ok, network}, network}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:define_network, definition}, _from, state) do
    case DefineNetwork.define_network(definition) do
      {:ok, network} ->
        {:reply, {:ok, network}, network}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:train, examples, expecteds, opts}, _from, network) do
    case Train.train(examples, expecteds, network, opts) do
      {:ok, new_network} ->
        {:reply, {:ok, new_network}, new_network}

      _ ->
        {:reply, {:error, "something went wrong"}, network}
    end
  end

  @impl true
  def handle_call({:train, examples, expecteds}, _from, network) do
    case Train.train(examples, expecteds, network, []) do
      {:ok, new_network} ->
        {:reply, {:ok, new_network}, new_network}

      _ ->
        {:reply, {:error, "something went wrong"}, network}
    end
  end

  # test_input is a list of input vectors to neural network
  # expected is a list of one hot encoded expected outputs
  # corresponding to the input list
  @impl true
  def handle_call({:test, test_inputs, expecteds}, _from, network) do
    # expected should be one hot encoded
    # each test input should be the right size to be input into the network
    results = Test.test(test_inputs, expecteds, network)

    {:reply, results, network}
  end

  @impl true
  def handle_call({:predict, input_vector}, _from, network) do
    prediction = Predict.prediction(input_vector, network)
    {:reply, prediction, network}
  end
end
