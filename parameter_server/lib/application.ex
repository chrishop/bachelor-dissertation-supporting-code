defmodule RingTMP.Application do
  use Application

  def start(_type, _args) do
    children = [
      {FeedForwardNetwork, []},
      {Task.Supervisor, name: RingTmp.TaskSupervisor}
    ]

    opts = [strategy: :one_for_one, name: RingTMP.Supervisor]

    Supervisor.start_link(children, opts)
  end
end
