defmodule MatrixUtilsTest do
  use ExUnit.Case

  describe "diag/1" do
    test "can diagonalise 1x1 matrix" do
      m =
        Matrex.new([
          [5]
        ])

      assert m == MatrixUtils.diag(m)
    end

    test "can diagonalise 2x1 matrix into 2x2" do
      m =
        Matrex.new([
          [5],
          [3]
        ])

      e =
        Matrex.new([
          [5, 0],
          [0, 3]
        ])

      assert e == MatrixUtils.diag(m)
    end

    test "can diagonalise 3x1 matrix into 3x3" do
      m =
        Matrex.new([
          [5],
          [3],
          [1]
        ])

      e =
        Matrex.new([
          [5, 0, 0],
          [0, 3, 0],
          [0, 0, 1]
        ])

      assert e == MatrixUtils.diag(m)
    end
  end
end
