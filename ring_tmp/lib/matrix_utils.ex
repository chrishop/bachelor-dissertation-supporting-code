defmodule MatrixUtils do
  def diag(matrix) do
    diag = Matrex.zeros(matrix[:rows])

    Matrex.column_to_list(matrix, 1)
    |> Enum.with_index()
    |> Enum.reduce(diag, fn {x, i}, diag -> diag |> Matrex.set(i + 1, i + 1, x) end)
  end
end
