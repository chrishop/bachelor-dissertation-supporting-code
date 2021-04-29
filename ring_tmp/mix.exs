defmodule RingTmp.MixProject do
  use Mix.Project

  def project do
    [
      app: :ring_tmp,
      version: "0.1.0",
      elixir: "~> 1.8",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      preferred_cli_env: [ run_node: :nodes, run_iris: :nodes, run_mnist: :nodes ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {RingTmp.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:matrex, "~> 0.6.8"},
      {:nimble_csv, "~> 1.0"}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
