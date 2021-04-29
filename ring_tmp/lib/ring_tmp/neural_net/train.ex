defmodule RingTmp.NeuralNet.Train do
  alias RingTmp.NeuralNet.Back

  def train(inputs, expecteds, network, opts) do
    Back.back(inputs, expecteds, network, opts)
    |> (fn x -> {:ok, x} end).()
  end
end
