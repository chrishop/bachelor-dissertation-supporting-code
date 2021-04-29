defmodule TestUtils do
  def matrix_equals(a, b, error \\ 0) do
    Matrex.subtract(a, b)
    |> Matrex.apply(:abs)
    |> Matrex.apply(fn x -> if x > error, do: 1.0, else: 0.0 end)
    |> Matrex.sum()
    |> case do
      0.0 -> true
      _ -> false
    end
  end
end
