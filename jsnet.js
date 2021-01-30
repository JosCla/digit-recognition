// neural network
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// revised as of 28 Dec. 2019

var output = [];
var input = [];
var totalInput;
var digit;

var canvas = document.getElementById('canv');
var ctx = canvas.getContext('2d');
var canvasRes = document.getElementById('results');
var ctxRes = canvasRes.getContext('2d');

function DigitRecog() {
    this.net = new Network([Math.pow(28, 2), 16, 16, 10]);
    this.pics, this.labels, this.totalPics, this.testInterval;
    this.successArr = [];
    this.costArr = [];
    this.avgNablaW = Array(this.net.numLayers-1).fill(0); 
    this.avgNablaB = Array(this.net.numLayers-1).fill(0);
    this.currTest = -1;
    this.totalSuccess = 0;
    this.avgCost = 0;
    
    // tests a certain picture to see how the network classifies it. can also display results
    this.test = function(pic, label, display, index) {
        var result = this.net.processInput(pic);        
        var desired = math.zeros(1, 10); desired['_data'][0][label] = 1;
        
        var verdict = maxIndex(result['_data'][0]);
        var cost = getCost(result, desired);
        
        if (display) {
            if (document.getElementById('disp-output-text').checked) {
                document.getElementById('output-text').innerHTML = 
                'Output: <br><br>' + 
                'Verdict: This is ' + verdict + '<br>Expected: This is ' + label + '<br>' + 
                (verdict === label ? '<span style="color: lime;"><b>Correct!</b></span>' : '<span style="color: red;"><b>Incorrect...</b></span>') + 
                '<br>Cost: ' + cost.toString().substr(0, 10);
            }
            
            if (document.getElementById('dispmode-one').checked) {
                drawNum(pic, label);
            } else if (document.getElementById('dispmode-all').checked) {
                drawNumSmall(pic, label, verdict, index);
            }
        }
        
        return [verdict, cost, desired];
    }
    
    this.testPlusBackprop = function(index) {
        // perform test
        var [verdict, cost, desired] = this.test(this.pics[index], this.labels[index], !document.getElementById('dispmode-none').checked, index);
        this.totalSuccess += (verdict === this.labels[index] ? 1 : 0);
        this.avgCost += cost/this.totalPics;
        
        // add backpropagation gradient to an average
        if (document.getElementById('learn').checked) {
            var [nablaW, nablaB] = this.net.backprop(desired);
            for (var i = 0; i < this.net.numLayers-1; i++) {
                this.avgNablaW[i] = math.add(this.avgNablaW[i], nablaW[i]);
                this.avgNablaB[i] = math.add(this.avgNablaB[i], nablaB[i]);
            }
        }
    }
    
    this.postTests = function() {
        // learn if needed
        if (document.getElementById('learn').checked) {
            for (var i = 0; i < this.net.numLayers-1; i++) {
                this.net.weights[i] = math.add(this.net.weights[i], math.multiply(this.avgNablaW[i], -document.getElementById('learn-speed').value/this.totalPics));
                this.net.biases[i] = math.add(this.net.biases[i], math.multiply(this.avgNablaB[i], -document.getElementById('learn-speed').value/this.totalPics));
            }
        }
        
        // logging success
        console.log('success: ' + this.totalSuccess);
        this.successArr.push(this.totalSuccess/this.totalPics);
        console.log('avg. cost: ' + this.avgCost);
        this.costArr.push(this.avgCost);
        
        // draw graph if needed
        if (document.getElementById('show-graph').checked) {
            drawGraph(this.successArr, this.costArr);
        }
        
        // output total success if needed
        if (document.getElementById('disp-output-text').checked) {
            document.getElementById('output-text').innerHTML = 'Output: <br><br>Total Success: ' + this.totalSuccess;
        }
        
        this.avgNablaW = Array(this.net.numLayers-1).fill(0); 
        this.avgNablaB = Array(this.net.numLayers-1).fill(0);
        this.totalSuccess = 0;
        this.avgCost = 0;
        this.getSet(parseInt(document.getElementById("index-size").value));
    }
    
    this.getSet = function(num) {
        [this.pics, this.labels] = getSet(num);
        this.totalPics = this.pics.length;
    }
    
    // this function is called by the setInterval, and as such its 'this' keyword would register as Window
    this.testAllInterval = function() {
        digit.testAll();
    }
    // tests all pictures quickly
    this.testAll = function() {
        if (this.testState === 0) {
            // setting test state
            this.testState = 1;
            
            // doing tests
            for (var i = 0; i < this.totalPics; i++) {
                this.testPlusBackprop(i);
            }
            
            this.postTests();
            
            // deciding whether or not to continue
            if (document.getElementById('continue').checked === false) {
                clearInterval(this.testInterval);
                this.testInterval = undefined;
            }
            
            // resetting test state
            this.testState = 0;
        }
    }
    
    // this function is called by the setInterval, and as such its 'this' keyword would register as Window
    this.testAllSlowInterval = function() {
        digit.testAllSlow();
    }
    // a slower but more visual version of testAll
    this.testAllSlow = function() {        
        if (this.currTest < this.totalPics) {
            this.testPlusBackprop(this.currTest);
        } else if (this.currTest === this.totalPics) {
            this.postTests();
            
            // check whether to continue
            if (document.getElementById('continue').checked === false) {
                clearInterval(this.testInterval);
                this.testInterval = undefined;
            }
            this.currTest = -1;
        }
        
        this.currTest++;
    }
    
    this.startDoTest = function() {        
        if (this.testInterval === undefined) {
            if (document.getElementById('speed-slow').checked) {
                this.testInterval = setInterval(this.testAllSlowInterval, 0.5);
                this.currTest = 0;
            } else if (document.getElementById('speed-fast').checked) {
                this.testInterval = setInterval(this.testAllInterval, 1);
                if (this.testState !== 1) {this.testState = 0;}
            }
            // resetting variables
            this.avgNablaW = Array(this.net.numLayers-1).fill(0); 
            this.avgNablaB = Array(this.net.numLayers-1).fill(0);
            this.totalSuccess = 0;
            this.avgCost = 0;
            this.getSet(parseInt(document.getElementById("index-size").value));
        }
    }
}

// initializes the neural network
function init() {
    if (digit !== undefined) {
        if (digit.testInterval !== undefined) {
            clearInterval(digit.testInterval);
        }
    }
    digit = new DigitRecog();
}

// selects a set from input and output arrays
function getSet(num) {
    var pics = []; var labels = [];
    for (var i = 0; i < num; i++) {
        var pos = Math.floor(Math.random() * totalInput);
        pics.push(math.multiply(input.slice(pos * Math.pow(28, 2), (pos+1) * Math.pow(28, 2)), 1/256));
        labels.push(output[pos]);
    }
    return [pics, labels];
}

// draws a number given pixel data and the label
function drawNum(pic, label) {
    ctx.clearRect(0, 0, 280, 280);
    for (var i = 0; i < pic.length; i++) {
        var currColor = (pic[i]*256).toString(16).padStart(2, '0');
        ctx.fillStyle = '#' + currColor + currColor + currColor;
        ctx.fillRect(10 * (i % 28), 10 * (Math.floor(i / 28)), 10, 10);
    }
    
    document.getElementById('label').textContent = 'Label: ' + label;
}

// draws a smaller number, used for the 'draw all' display option
function drawNumSmall(pic, label, verdict, index) {
    for (var i = 0; i < pic.length; i++) {
        var colStr = (pic[i]*256).toString(16).padStart(2, '0');
        
        ctx.fillStyle = (verdict === label ? "#00" + colStr + "00" : "#" + colStr + "0000");
        ctx.fillRect(28 * ((index % 100) % 10) + (i % 28), 28 * Math.floor((index % 100) / 10) + Math.floor(i / 28), 1, 1);
    }
    
    document.getElementById('label').textContent = 'Label: ' + label;
}

// draws a graph showing progression of cost/success
function drawGraph(successArr, costArr) {
    var successes = successArr.slice(Math.max(0, successArr.length-100), successArr.length);
    var costs = costArr.slice(Math.max(0, costArr.length-100), costArr.length);
    var width = canvasRes.width;
    var height = canvasRes.height;
    ctxRes.clearRect(0, 0, width, height);
    
    // axes
    for (var i = 0; i < 5; i++) {
        ctxRes.beginPath();
        ctxRes.strokeStyle = '#000000';
        ctxRes.moveTo(0, i*(height/5));
        ctxRes.lineTo(width, i*(height/5));
        ctxRes.stroke();
        
        ctxRes.fillStyle = '#0000FF';
        ctxRes.textAlign = 'right';
        ctxRes.fillText(Math.round((1-(i/5))*100).toString() + '%', width, i*(height/5));
        
        ctxRes.fillStyle = '#FF0000';
        ctxRes.textAlign = 'left';
        ctxRes.fillText((Math.round((1-(i/5))*100)/100).toString(), 0, i*(height/5));
    }
    
    // success rate
    ctxRes.beginPath();
    ctxRes.strokeStyle = '#0000FF';
    ctxRes.moveTo(0, height * (1-successes[0]));
    for (var i = 1; i < successes.length; i++) {
        ctxRes.lineTo((width/100) * i, height * (1-successes[i]));
    }
    ctxRes.stroke();
    
    // cost
    ctxRes.beginPath();
    ctxRes.strokeStyle = '#FF0000';
    ctxRes.moveTo(0, height * (1-costs[0]));
    for (var i = 1; i < costs.length; i++) {
        ctxRes.lineTo((width/100) * i, height * (1-costs[i]));
    }
    ctxRes.stroke();
}

// returns index of max value in array
function maxIndex(arr) {
    var max = -Infinity; var ind = -1;
    for (var i in arr) {if (arr[i] > max) {max = arr[i]; ind = i;}}
    return parseInt(ind);
}

// getting and receiving text files
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
var ready = 0;

// getting input
var request = new XMLHttpRequest();
request.responseType = 'arraybuffer';
request.open('GET', 'train-images.idx3-ubyte', true);
request.onreadystatechange = function() {
    
    if (this.readyState === 4) {
        
        var arrayBuffer = this.response;
        var byteArr = new Uint8Array(arrayBuffer);
        
        input = [];
        for (var i = 0; i < byteArr.length; i++) {input.push(byteArr[i]);}
        input.splice(0, 16);
        
        console.log('Request complete. (input)');
        ready++;
        
        document.getElementById('loading-input').innerHTML += 'Complete!';
        if (ready === 2) {readyUp();}
        
    }
    
}
request.send();

// getting output
var request = new XMLHttpRequest();
request.responseType = 'arraybuffer';
request.open('GET', 'train-labels.idx1-ubyte', true);
request.onreadystatechange = function() {
    
    if (this.readyState === 4) {
        
        var arrayBuffer = this.response;
        var byteArr = new Uint8Array(arrayBuffer);
        
        for (var i = 0; i < byteArr.length; i++) {output.push(byteArr[i]);}
        
        output.splice(0, 4);
        var temp = output.splice(0, 4);
        totalInput = ((temp[3]) + (Math.pow(2, 8) * temp[2]) + (Math.pow(2, 16) * temp[1]) + (Math.pow(2, 24) * temp[0]));
        
        console.log('Request complete. (output)');
        ready++;
        
        document.getElementById('loading-output').innerHTML += 'Complete!';
        if (ready === 2) {readyUp();}
        
    }
    
}
request.send();

function readyUp() {
    document.getElementById('main').style.display = '';
    document.getElementById('loading').style.display = 'none';
    init();
}

// changing tabs in UI
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function changeTab(tab) {
    document.getElementById('open-test').classList.remove('selected');
    document.getElementById('open-settings').classList.remove('selected');
    
    document.getElementById('open-' + tab).classList.add('selected');
    
    document.getElementById('div-test').style.display = 'none';
    document.getElementById('div-settings').style.display = 'none';
    
    document.getElementById('div-' + tab).style.display = '';
}