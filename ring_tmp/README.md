# RingTmp
A Ring Topological Model Parallel distributed neural network framework

## Installation
### Mac
Should work out of the box but if you get this error:
```
native/src/matrix_dot.c:5:10: fatal error: 'cblas.h' file not found
```

make sure x code command line tools are installed:
```
xcode-select --install
```
If that doesn't resolve you can find more command line options
[here](https://github.com/versilov/matrex)

In the same level as the `mix.exs`
```
mix deps.get
```

### Ubuntu
Install dependencies:
```
sudo apt-get install build-essential erlang-dev libatlas-base-dev
```

In the same level as the `mix.exs`
```
mix deps.get
```

### Windows
Doesn't support Windows, as one of the libraries isn't compatible with the
platform.

## Usage
To train on the Iris dataset:
Terminal 1:
```
mix iris_2_2
```

Terminal 2:
```
mix iris_1_2
```



To train on the MNIST dataset:
Terminal 1:
```
mix mnist_2_2
```

Terminal 2:
```
mix mnist_1_2
```