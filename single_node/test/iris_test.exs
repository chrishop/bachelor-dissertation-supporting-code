defmodule IrisTest do
  use ExUnit.Case

  describe "train_test_data/0" do
    test "should split data correctly" do

      {train_data, test_data} = Iris.train_test_data()
      assert length(train_data) == 150
      assert length(test_data) == 0

      {train_data, test_data} = Iris.train_test_data(0.25)
      assert length(train_data) == 38
      assert length(test_data) == 112

      {train_data, test_data} = Iris.train_test_data(0.5)
      assert length(train_data) == 75
      assert length(test_data) == 75


      {train_data, test_data} = Iris.train_test_data(0)
      assert length(train_data) == 0
      assert length(test_data) == 150
    end
  end

  describe "get_features/0" do
    test "has correct row" do
      features = Iris.features()

      assert List.first(features) == Matrex.new([[5.1], [3.5], [1.4], [0.2]])
      assert length(features) == 150
    end
  end

  describe "get_binary_expecteds" do
    test "with 'setosa' has correct truth values" do
      expecteds = Iris.get_binary_expecteds("setosa")

      assert {150, 1} == Matrex.size(expecteds)
      assert expecteds[1] == 1.0
    end
  end

  describe "one_hot_encoded" do
    test "as expected" do
      targets = Iris.one_hot_encoded()

      assert List.first(targets) == Matrex.new([[1], [0], [0]])
      assert length(targets) == 150
    end
  end

end
