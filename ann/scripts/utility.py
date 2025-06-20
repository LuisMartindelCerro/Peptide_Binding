### Helper Functions
import numpy as np

## Encode
def encode(peptides, encoding_scheme, alphabet):
    encoded_peptides = []

    for peptide in peptides:
        encoded_peptide = []
        for peptide_letter in peptide:
            for alphabet_letter in alphabet:
                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])
        
        # add a 1 (bias)
        encoded_peptide.append(1)
        
        # store peptide
        encoded_peptides.append(encoded_peptide)
    return np.array(encoded_peptides)

## Signoid Handling
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(a):
    return (1-a)*a

## Error
def error(y, y_pred):
    return 0.5*(y_pred - y)**2

## Propagation
    # Forward
def forward(X, w1, w2):
   
    # get dimension, substracting the bias
    input_layer_dim = w1.shape[0] - 1 
    hidden_layer_dim = w2.shape[0] - 1
    
    ################
    # hidden layer #
    ################
    
    # activity of hidden layer
    for j in range(hidden_layer_dim):
        z = 0.0
        for i in range(input_layer_dim+1):
            z += X[0][i]* w1[i,j]
        X[1][j] = sigmoid(z)
    
    ################
    # output layer #
    ################
    
    z = 0
    for i in range(hidden_layer_dim+1):
        z += X[1][i]*w2[i,0]
    X[2][0] = sigmoid(z)

    # Backward
def back_prop(X, t, w_h, w_o, dj_dw_h, dj_dw_o):
    # dj_dw_h are the derivativs with respect to the weights
    # connecting the input to the hidden layer
    # dj_dw_o are the derivatics with respect to the weights
    # connecting the hidden to the outout layer

    # get dimension, substracting the bias
    input_layer_dim = w_h.shape[0] - 1 
    hidden_layer_dim = w_o.shape[0] - 1
    
    ##############################################    
    # derivative of cost function respect to w_o #
    # Remember X[2][0] is the prediction value,
    # And dj_dw_o = dE/dw = dE/dO * dO/do * do/dw
    ##############################################
    
    delta = (X[2][0] - t) * X[2][0] * (1 - X[2][0])
    
    for i in range(hidden_layer_dim+1):
        dj_dw_o[i] = delta * X[1][i] 
    
    ##############################################
    # derivative of cost function respect to w_h #
    # Remember dj_dw_h = dE/dv where v are the weight connecting
    # the input to the hidden layer, and
    # dE/dv = dE/dO * dO/do * do/dH * dH/dh * dh/dv
    # where H is the output from hidden neuron j, and v is the
    # weight connecting input neuron i to hidden neuron j
    ##############################################
    
    for j in range(hidden_layer_dim):
        delta2 = delta * w_o[j, 0] * X[1][j] * (1 - X[1][j]) 
        for i in range (input_layer_dim+1): # +1 to include the input layer bias
            dj_dw_h[i, j] = delta2 * X[0][i] 

## Network Handling
    # Architecture (Feed Forward)
def feed_forward_network(input_layer_dim, hidden_layer_dim, output_layer_dim):
    # layer dimensions
    i_dim = input_layer_dim      # vector of shape (i_dim,) 
    h_dim = hidden_layer_dim     # matrix of shape (i_dim, h_dim)
    o_dim = output_layer_dim     # matrix of shape (h_dim, o_dim)
    
    # hidden layer weights
    # w_h[i, j] is the weight that links input's feature "i" to neuron "j" of the hidden layer
    w_h = np.random.uniform(-0.1, 0.1, size=(i_dim+1)*h_dim).reshape(i_dim+1, h_dim)
        
    # output layer weights
    # w_o[i, j] is the weight that links hidden layer's neuron "i" to neuron "j" of the output layer
    # since we only have one output neuron, j = 1, and w_o behaves as a vector, not a matrix
    w_o = np.random.uniform(-0.1, 0.1, size=(h_dim+1)*o_dim).reshape(h_dim+1, o_dim)
    
    # X matrix, X stores the output from each layer
    X_dim = max(i_dim, h_dim, o_dim)
    X = np.zeros(shape=(3, X_dim+1))
    
    # The last column of the X layer is one, to deal with the bias
    X[0][input_layer_dim] = 1.0 
    X[1][hidden_layer_dim] = 1.0
    
    # print network summary
    print("NETWORK SUMMARY")
    print("---------------" )
    print("Input Layer shape:", (1, input_layer_dim))
    print("Hidden Layer shape:", w_h.shape)
    print("Output layer shape:", w_o.shape)
    print("Total parameters:", (i_dim+1)*h_dim + (h_dim+1)*o_dim)
    print("")
    
    # return everything
    return w_h, w_o, X

    # Save
def save_network(file_name, w_h, w_o, lpcc, lerr, tpcc, terr, epochs):
    input_layer_dim = w_h.shape[0]
    hidden_layer_dim = w_o.shape[0]
    output_layer_dim = w_o.shape[1]

    with open(file_name, 'w') as file:
        # run data
        file.write("TESTRUNID")
        file.write(" EPOCH: " + str(epochs))
        file.write(" L_PCC: " + str(lpcc))
        file.write(" L_ERR: " + str(lerr))
        file.write(" T_PCC: " + str(tpcc))
        file.write(" T_ERR: " + str(terr))
        file.write("\n")

        # LAYER: 1 
        file.write(str(input_layer_dim-1) + " LAYER: 1")
        file.write("\n")

        # LAYER: 2 
        file.write(str(hidden_layer_dim-1) + " LAYER: 2")
        file.write("\n")

        # LAYER: 3
        file.write(str(output_layer_dim) + " LAYER: 3")
        file.write("\n")

        # number of training cycles
        # :ILEARN 
        file.write(str(epochs) + " :ILEARN")
        file.write("\n")
        
        # network weights (five per line)
        weights = [w_h, w_o]
        
        cnt = 0

        for w in weights:
            w = w.flatten()
            for i in range(0, len(w)):
                file.write(str(w[i]) + str("\t"))
                cnt += 1

                if cnt == 5:
                    file.write("\n")
                    cnt = 0   
        if cnt != 0:
            file.write("\n")

    # Load
def load_network(file_name):
    f = open(file_name, "r")
    n_line = 0
    weight_list = []

    for line in f:
        # clean and separate line
        sline = line.strip().split()

        # input layer dimension
        if n_line == 1:
            input_layer_dim = int(sline[0])

        # hidden layer dimension    
        if n_line == 2:
            hidden_layer_dim = int(sline[0])

        # output layer dimension
        if n_line == 3:
            output_layer_dim = int(sline[0])

        # model weights
        if n_line >= 5:
            for i in range(0, len(sline)):
                weight_list.append(float(sline[i]))

        n_line += 1

    # HIDDEN LAYER WEIGHTS
    # w_h[i, j] is the weight that links input's feature "i" to neuron "j" of the hidden layer        
    w_h_load = np.zeros(shape=(input_layer_dim+1, hidden_layer_dim))

    for i in range(0, (input_layer_dim+1)*hidden_layer_dim, hidden_layer_dim):
        for j in range(0, hidden_layer_dim):
            row = i // hidden_layer_dim
            w_h_load[row, j] = weight_list[i+j]
      
    # OUTPUT LAYER WEIGHTS
    # w_o[i, j] is the weight that links hidden layer's neuron "i" to neuron "j" of the output layer
    w_o_load = np.zeros(shape=(hidden_layer_dim+1, output_layer_dim))
    w_h_end = (input_layer_dim+1) * hidden_layer_dim

    for i in range(w_h_end, w_h_end+hidden_layer_dim+1, output_layer_dim):
        for j in range(0, output_layer_dim):
            row = (i - w_h_end) // output_layer_dim
            w_o_load[row, j] = weight_list[i+j]
                
    # return weight matrices
    return w_h_load, w_o_load