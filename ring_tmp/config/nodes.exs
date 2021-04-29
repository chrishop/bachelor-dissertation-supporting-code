use Mix.Config


config :ring_tmp,

  iris_1_1: %{
    app_node: %{
      node_type: :master,
      master_node: :alice@localhost,
      this_node: :alice@localhost,
      prev_node: :alice@localhost,
      next_node: :alice@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 4, 3}]
    }
  },

  iris_1_2: %{
    app_node: %{
      node_type: :master,
      master_node: :alice@localhost,
      this_node: :alice@localhost,
      prev_node: :bob@localhost,
      next_node: :bob@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 4, 4}],
      opts: [learning_rate: 0.005]
    }
  },
  iris_2_2: %{
    app_node: %{
      node_type: :worker,
      master_node: :alice@localhost,
      this_node: :bob@localhost,
      prev_node: :alice@localhost,
      next_node: :alice@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 4, 3}],
      opts: [learning_rate: 0.005]
    }
  },


  iris_1_3: %{
    app_node: %{
      node_type: :master,
      master_node: :alice@localhost,
      this_node: :alice@localhost,
      prev_node: :chris@localhost,
      next_node: :bob@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 4, 8}],
      opts: [learning_rate: 0.0005]
    }
  },

  iris_2_3: %{
    app_node: %{
      node_type: :worker,
      master_node: :alice@localhost,
      this_node: :bob@localhost,
      prev_node: :alice@localhost,
      next_node: :chris@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 8, 8}],
      opts: [learning_rate: 0.0005]
    }
  },
  iris_3_3: %{
    app_node: %{
      node_type: :worker,
      master_node: :alice@localhost,
      this_node: :chris@localhost,
      prev_node: :bob@localhost,
      next_node: :alice@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 8, 3}],
      opts: [learning_rate: 0.0005]
    }
  },



  mnist_1_2: %{
    app_node: %{
      node_type: :master,
      master_node: :alice@localhost,
      this_node: :alice@localhost,
      prev_node: :bob@localhost,
      next_node: :bob@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 784, 50}],
      opts: [learning_rate: 0.0005]
    }
  },
  mnist_2_2: %{
    app_node: %{
      node_type: :worker,
      master_node: :alice@localhost,
      this_node: :bob@localhost,
      prev_node: :alice@localhost,
      next_node: :alice@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 50, 10}],
      opts: [learning_rate: 0.0005]
    }
  },

  scale_1_4: %{
    app_node: %{
      node_type: :master,
      master_node: :a@localhost,
      this_node: :a@localhost,
      prev_node: :d@localhost,
      next_node: :b@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 784, 50},{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_2_4: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :b@localhost,
      prev_node: :a@localhost,
      next_node: :c@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_3_4: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :c@localhost,
      prev_node: :b@localhost,
      next_node: :d@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_4_4: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :d@localhost,
      prev_node: :c@localhost,
      next_node: :a@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}, {:output_layer, 50, 10}],
      opts: [learning_rate: 0.001]
    }
  },


  scale_1_6: %{
    app_node: %{
      node_type: :master,
      master_node: :a@localhost,
      this_node: :a@localhost,
      prev_node: :f@localhost,
      next_node: :b@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 784, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_2_6: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :b@localhost,
      prev_node: :a@localhost,
      next_node: :c@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_3_6: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :c@localhost,
      prev_node: :b@localhost,
      next_node: :d@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_4_6: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :d@localhost,
      prev_node: :c@localhost,
      next_node: :e@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_5_6: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :e@localhost,
      prev_node: :d@localhost,
      next_node: :f@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_6_6: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :f@localhost,
      prev_node: :e@localhost,
      next_node: :a@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:output_layer, 50, 10}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_1_8: %{
    app_node: %{
      node_type: :master,
      master_node: :a@localhost,
      this_node: :a@localhost,
      prev_node: :h@localhost,
      next_node: :b@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 784, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_2_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :b@localhost,
      prev_node: :a@localhost,
      next_node: :c@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_3_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :c@localhost,
      prev_node: :b@localhost,
      next_node: :d@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_4_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :d@localhost,
      prev_node: :c@localhost,
      next_node: :e@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_5_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :e@localhost,
      prev_node: :d@localhost,
      next_node: :f@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_6_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :f@localhost,
      prev_node: :e@localhost,
      next_node: :g@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_7_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :g@localhost,
      prev_node: :f@localhost,
      next_node: :h@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.001]
    }
  },

  scale_8_8: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :h@localhost,
      prev_node: :g@localhost,
      next_node: :a@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}, {:output_layer, 50, 10}],
      opts: [learning_rate: 0.001]
    }
  },


  scale_1_10: %{
    app_node: %{
      node_type: :master,
      master_node: :a@localhost,
      this_node: :a@localhost,
      prev_node: :j@localhost,
      next_node: :b@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 784, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_2_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :b@localhost,
      prev_node: :a@localhost,
      next_node: :c@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_3_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :c@localhost,
      prev_node: :b@localhost,
      next_node: :d@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_4_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :d@localhost,
      prev_node: :c@localhost,
      next_node: :e@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_5_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :e@localhost,
      prev_node: :d@localhost,
      next_node: :f@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_6_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :f@localhost,
      prev_node: :e@localhost,
      next_node: :g@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_7_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :g@localhost,
      prev_node: :f@localhost,
      next_node: :h@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_8_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :h@localhost,
      prev_node: :g@localhost,
      next_node: :i@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_9_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :i@localhost,
      prev_node: :h@localhost,
      next_node: :j@localhost
    },
    neural_net: %{
      definition: [{:hidden_layer, 50, 50}],
      opts: [learning_rate: 0.000]
    }
  },

  scale_10_10: %{
    app_node: %{
      node_type: :worker,
      master_node: :a@localhost,
      this_node: :j@localhost,
      prev_node: :i@localhost,
      next_node: :a@localhost
    },
    neural_net: %{
      definition: [{:output_layer, 50, 10}],
      opts: [learning_rate: 0.000]
    }
  }
