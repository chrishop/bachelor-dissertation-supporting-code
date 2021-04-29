defmodule RingTmp.MixProject do
  use Mix.Project

  def project do
    [
      app: :ring_tmp,
      version: "0.1.0",
      elixir: "~> 1.8",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {RingTMP.Application, []}
    ]
  end

  defp aliases() do
    [
      unit_test: &unit_test/1,
      e2e_test: &e2e_test/1
    ]
  end

  defp unit_test(_) do
    Mix.env(:test)
    Mix.Task.run("test", ["--exclude", "end_to_end:true"])
  end

  defp e2e_test(_) do
    Mix.env(:test)
    Mix.Task.run("test", ["--only", "end_to_end:true"])
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:gen_stage, "~> 1.0"},
      {:matrex, "~> 0.6.8"},
      {:nimble_csv, "~> 1.0"},
      {:poolboy, "~> 1.5"}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
