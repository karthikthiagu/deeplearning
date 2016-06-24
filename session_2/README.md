
-> Dependencies:
	wget: sudo apt-get install wget
	CAFFE_ROOT: ./data/cifar10/get_cifar10.sh
		    ./examples/cifar10/create_cifar10.sh

-> The solver mode has been set to CPU in all the solver files.

-> To train model: ./examples/cifar10/train_quick.sh - from CAFFE_ROOT

-> To train the model with the cifar10_solver.prototxt: ./build/tools/caffe train --solver=/path/to/solver/ - from CAFFE_ROOT

