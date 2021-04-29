defmodule FeedForwardNetwork.Train do
  alias FeedForwardNetwork.Back

  def train(inputs, expecteds, network, opts) do
    Back.back(inputs, expecteds, network, opts)
    |> (fn x -> {:ok, x} end).()
  end
end
