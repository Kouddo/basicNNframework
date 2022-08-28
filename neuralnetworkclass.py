import numpy as np

#made using MNIST dataset slightly easier
#from mnist.loader MNIST 



class neuralnetwork(object):
    def __init__(self):
        self.layer = []
        self.bias = []

        
        
    #forward propogation between layers, needs to be ran once
    def forwardprop(indata):
        for i,x in (enumerate(layer)):
            indata = (relu(np.matmul(indata, layer[i])+bias[i]))
        return indata
    
    #activation functions
    def relu(x):
        return(np.maximum(0,x))

    def backrelu(dx, y):
        dy = np.array(dx, copy = True)
        dy[y <= 0] = 0;
        return(dy)

    def sigmoid(x):
        return(1/(1+np.exp(-x)))

    
    
    #definiting the network initially so training can begin
    #network dimension = [[input layer size], [hiddenlayer1 dimensions], [hiddenlayer2 dimensions], ... [output layer dimensions]]
    def initializenetwork(networkdimension):
        layer = []
        bias = []
        layer.append(np.random.rand(networkdimension[0],networkdimension[1]))
        bias.append(np.random.rand(networkdimension[1]))
        for i in range(len(networkdimension)-3):
            layer.append(np.random.rand(networkdimension[i+1], networkdimension[i+2]))
            bias.append(np.random.rand(networkdimension[i+2]))
        layer.append(np.random.rand(networkdimension[len(networkdimension)-2],networkdimension[len(networkdimension)-1]))
        bias.append(np.random.rand(networkdimension[len(networkdimension)-1]))
        return(layer, bias)
    
    
    def softmax(inputdata, correct):
        summa = []
        for i in range(len(inputdata)):
            summa.append(np.exp(inputdata[i]))
            
        return(np.exp(correct) / (sum(summa)))
    
    
    #testing accuracy of model, outputs % corrent
    def runtest(testimage, testlabel, iterations):
        count = 0
        for i in range(iterations):
            x = neuralnetwork.forwardprop(testimage[i])
            if(np.argmax(x) == testlabel[i]):
                count = count+1
        return (count/iterations)*100



