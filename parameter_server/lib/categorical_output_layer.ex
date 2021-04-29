defmodule CategoricalOutputLayer do
  alias MatrixUtils

  @moduledoc """
  This output layer has one fully connected layer, after which the softmax
  function is applied. The sigmoid is not applied before entering softmax
  layer as the softmax layer *is* an activation function.
  """

  @doc ~S"""
  calculates the activation of a dense layer being put into a softmax layer.

  ## Parameters
    - prev_activation: activation of the previous layer
    - weights: weights of the current layer
    - bias: redundant for this layer
    - opts: redundant for this layer

  ## Returns
    layer activation
  """
  def forward(prev_activation, weights, bias, _opts \\ []) do
    Matrex.dot_tn(weights, prev_activation)
    |> Matrex.add(bias)
    |> softmax()
  end

  @doc ~S"""
  calculates the change in the weights and the bias as well as passing error
  back to other layers.

  ## Parameters
    - activation: The activation vector which is the output of the softmax layer
    - weights: The weights matrix of the fully connected layer within the softmax layer
    - bias: Is ignored here as it had no effect on the softmax activation function
    - target_activation: the expected activation vector of the network, given by the training pairs
    - opts options include `:learning_rate`

  ## Returns
    `{new_weights, new_bias, remaining_error}`

    remaining_error is the cost with respect to the activations of the previous layer.


  """
  def back(activation, prev_activation, weights, bias, target_activation, opts \\ []) do
    cost_wrt_z = cost_wrt_z(target_activation, activation)

    weight_delta =
      Matrex.dot_nt(prev_activation, cost_wrt_z)
      |> Matrex.multiply(Keyword.get(opts, :learning_rate, 1.0))

    bias_delta = cost_wrt_z |> Matrex.multiply(Keyword.get(opts, :learning_rate, 1.0))

    remaining_error = Matrex.dot(weights, cost_wrt_z)

    new_weights = Matrex.add(weights, weight_delta, 1, 1)
    new_bias = Matrex.add(bias, bias_delta)
    {new_weights, new_bias, remaining_error}
  end

  @doc """
  This implements cross categorical entropy loss function.

  ## Parameters
    - activations: vector of activations, the output of the forward function of this layer
    - target_activations: the expected output of the layer as given by the input-ouput training pairs

  ## Returns
    loss value as a float
  """
  def loss(activation, target_activation) do
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

  defp softmax(z_vec) do
    stabilised_vec = Matrex.subtract(z_vec, Matrex.max(z_vec))
    exp = Matrex.apply(stabilised_vec, :exp)

    Matrex.divide(exp, Matrex.sum(exp))
  end

  # derivative of the cost function wrt the softmax function
  defp cost_wrt_z(softmax_activation, target_activation) do
    Matrex.subtract(softmax_activation, target_activation)
  end

  # derivative of z with respect to weights
  defp z_wrt_w(prev_activation) do
    Matrex.transpose(prev_activation)
  end

  # derivative of z with respect to previous layer activation
  defp z_wrt_prev_a(weights) do
    Matrex.transpose(weights)
  end


  defp get_target_category(target_activation) do
    {row, _col} = Matrex.size(target_activation)

    Enum.map(1..row, fn x -> [x] end)
    |> Matrex.new()
    |> Matrex.multiply(target_activation)
    |> Matrex.sum()
    |> round()
  end

end
