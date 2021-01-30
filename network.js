// Neural Network
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Note: mathjs must be included to use this program.
// Last updated 28 Dec. 2019. Made by JosCla.

// Creates a basic multilayer perceptron. (Fully Connected Layer)
function Network(sizes) {
    this.sizes = sizes;
    this.numLayers = this.sizes.length;
    this.biases = Array(this.numLayers-1);
        for (var i = 0; i < this.biases.length; i++) {
            this.biases[i] = math.zeros(1, this.sizes[i+1]);
            this.biases[i]['_data'] = math.random([1, this.sizes[i+1]], -1, 1);
        }
    this.weights = Array(this.numLayers-1);
        for (var i = 0; i < this.weights.length; i++) {
            this.weights[i] = math.zeros(this.sizes[i], this.sizes[i+1]);
            this.weights[i]['_data'] = math.random([this.sizes[i], this.sizes[i+1]], -1, 1);
        }
    
    this.zs = Array(this.numLayers);
    this.activations = Array(this.numLayers);
    
    // return sig(Aw+b);
    this.feedForward = function(layer, input) {
        var ret = math.add(math.multiply(input, this.weights[layer]), this.biases[layer])
        return [ret, sig(ret)];
    }
    
    // feeds an input through each layer of the network, returns the output layer
    this.processInput = function(input) {
        if (!input.isMatrix) {input = math.matrix(input);}
        if (input.sizes !== [1, 2]) {input.reshape([1, this.sizes[0]]);}
        this.activations[0] = (input);
        for (var i = 0; i < (this.numLayers-1); i++) {
            [this.zs[i+1], this.activations[i+1]] = this.feedForward(i, input);
            input = this.activations[i+1];
        }
        return input;
    }
    
    // uses backpropagation to find gradient with respect to weights and biases
    // code inspired by https://github.com/mnielsen/neural-networks-and-deep-learning
    this.backprop = function(desired) {
        var nablaB = Array(this.numLayers-1);
        var nablaW = Array(this.numLayers-1);
        var nablaInit = math.subtract(this.activations[this.numLayers-1], desired);
        var delta = math.dotMultiply(nablaInit, sig(this.zs[this.numLayers-1], true));
        
        for (var i = this.numLayers-2; i >= 0; i--) {
            nablaW[i] = math.multiply(math.transpose(this.activations[i]), delta);
            nablaB[i] = delta;
            if (i !== 0) {delta = math.dotMultiply(math.multiply(delta, math.transpose(this.weights[i])), sig(this.zs[i], true));}
        }
        return [nablaW, nablaB];
    }
    
    // randomly mutates the values of the network, given a max change value
    this.mutate = function(maxChange) {
        for (var i = 0; i < this.biases.length; i++) {
            var change = math.random([1, this.sizes[i+1]], -maxChange, maxChange);
            this.biases[i] = math.add(this.biases[i], change);
        }
        for (var i = 0; i < this.weights.length; i++) {
            var change = math.random([this.sizes[i], this.sizes[i+1]], -maxChange, maxChange);
            this.weights[i] = math.add(this.weights[i], change);
        }
    }
    
    // returns a dereferenced copy of the network
    this.clone = function() {
        var netClone = new Network(this.sizes);
        for (var i = 0; i < netClone.biases.length; i++) {
            netClone.biases[i] = this.biases[i].clone();
        }
        for (var i = 0; i < netClone.weights.length; i++) {
            netClone.weights[i] = this.weights[i].clone();
        }
        return netClone;
    }
}

// sigmoid function with support for arrays and matrices too. prime flag indicates sigmoid prime
function sig(n, prime) {
    if (n.isMatrix) {
        var ret = math.matrix(math.zeros(n['_size']));
        ret['_data'] = sig(n['_data'], prime);
        return ret;
    } else if (typeof(n) === 'object') {
        var ret = [];
        for (var i in n) {ret[i] = sig(n[i], prime);}
        return ret;
    } else if (typeof(n) === 'number') {
        return (prime ? sigPFunc(n) : sigFunc(n));
    } else {
        console.log('Invalid type for sigmoid.');
    }
}

// sigmoid function
function sigFunc(n) {
    return (1 / (1 + Math.pow(Math.E, -1 * n)));
}

// sigmoid prime function
function sigPFunc(n) {
    return sigFunc(n) * (1-sigFunc(n));
}

// gets the sum of all elements
function getSum(n) {
    var total = 0;
    if (n.isMatrix) {
        return getSum(n['_data']);
    } else if (typeof(n) === 'object') {
        for (var i in n) {total += getSum(n[i]);}
        return total;
    } else if (typeof(n) === 'number') {
        return n;
    } else {
        console.log('Invalid type for summation.');
    }
}

// given actual and desired activations, returns the cost
function getCost(actual, desired) {
    var cost = math.subtract(desired, actual);
    cost = math.square(cost);
    return getSum(cost);
}