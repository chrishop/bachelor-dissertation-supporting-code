defmodule Mix.Tasks.ServerWorker do
  use Mix.Task

  def run([name]) do
    Mix.Task.run("app.start")

    hostname = String.to_atom(name)
    Node.start(hostname, :shortnames)

    IO.puts("Worker is online, press enter to exit:")
    IO.read(:stdio, :line)

  end

end
