import numpy as np

# Takes a 28x28 array x, as well as a list of weight matrices (the ws) and bias vectors (the #bs), and computes the forward pass of a neural network with these parameters.

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def softmax(z):
  exps = np.exp(z)
  return exps/(np.sum(exps))

def forward(x, ws, bs):
  layer_outputs = []
  x = np.reshape(x, (784,))
  for i in range(len(ws)):
    layer_outputs.append(x)
    u = np.dot(ws[i], x) + bs[i]
    layer_outputs.append(u)
    x = sigmoid(u)
  layer_outputs.append(softmax(u))
  return layer_outputs

# y = prediction, t = target

def loss(y, t):
  return -np.log(y[t])

# Now want dLoss/dparameters

def dlossdws_and_bs(layer_outputs, ws, bs, t):

# Compute dloss by du in a vector, where the ith element is dloss by dui.

  dloss_du_i = layer_outputs[-1]
  dloss_du_i[t] = dloss_du_i[t] - 1

 # Now write a for loop that does the below.

  dloss_dws = []
  dloss_dbs = []
  layer = len(layer_outputs)-2
 # Can loop through the ws/bs
  for k in reversed(range(len(ws))):
   # compute a matrix of dloss by dws
    u = layer_outputs[layer]
    x = layer_outputs[layer-1]
    p = x.reshape((1,-1))
    q = dloss_du_i.reshape((-1,1))
    #print np.shape(q)
    dloss_dws.insert(0, np.dot(q, p))
    dloss_dbs.insert(0, dloss_du_i)
    dloss_dx_i = np.dot(np.transpose(ws[k]),dloss_du_i)
    dloss_du_i = dloss_dx_i*x*(1-x)
    layer -= 2
  return dloss_dws, dloss_dbs

num_hidden_1 = 50
num_hidden_2 = 20
num_output = 10

x = np.random.randn(28,28)
w1 = np.random.randn(num_hidden_1, 784)
w2 = np.random.randn(num_hidden_2, num_hidden_1)
w3 = np.random.randn(num_output, num_hidden_2)
ws = [w1,w2,w3]
b1 = np.random.randn(num_hidden_1)
b2 = np.random.randn(num_hidden_2)
b3 = np.random.randn(num_output)
bs = [b1,b2,b3]
t = 7
# print forward(x, ws, bs)

out = forward(x, ws, bs)

e = 1e-5

gws, gbs = dlossdws_and_bs(out, ws, bs, t)

def test_grad_2(x, ws, bs):
  gbts = []

  for k in range(len(bs)):
    gbt = np.zeros(np.shape(bs[k]))

    for i in range(len(bs[k])):
      bc = np.copy(bs[k])
      bc[i] += e
      bsc = np.copy(bs)
      bsc[k] = bc
      loss_plus = loss(forward(x, ws, bsc)[-1], t)
      bc[i] -= 2*e
      loss_minus = loss(forward(x, ws, bsc)[-1], t)
      gbt[i] = (loss_plus - loss_minus)/(2*e)

    gbts.append(gbt)

  return gbts

gbts = test_grad_2(x, ws, bs)

for i in range(len(bs)):

  comp = gbs[i] - gbts[i]

  print np.max(np.abs(comp))

  print np.max(np.abs(comp)/(e + np.abs(gbs[i]) + np.abs(gbts[i])))


# Returns the accuracy, i.e. proportion correct.
def accuracy(predictions, Y):
  correct = 0.0

  for i in range(len(predictions)):
    if predictions[i] == Y[i]:
      correct += 1.0
  percent = float(100.0 * correct) / float(len(predictions))
  return percent


num_examples = 50

# X is a collection of 28x28 'images' -- random matrices
X = np.random.randn(num_examples,28,28)

# Y is a label 0-9 for each image
Y = np.random.choice(10, num_examples)

def get_outputs(X, Y, ws, bs):
  num_examples = np.shape(X)[0]
  outputs = []
  predictions = []
  losses = []

  for i in range (num_examples):

    # Compute the output of X[i,:,:]
    output = forward(X[i,:,:], ws, bs)[-1]
    outputs.append(output)

    # Compute the prediction of X[i,:,:]
    prediction = np.argmax(output)
    predictions.append(prediction)

    # Compute the loss
    loss_i = loss(output, Y[i])
    losses.append(loss_i)
    #print i

  return outputs, predictions, losses


#print get_outputs(X, Y, ws, bs)


# Stochastic gradient descent. Take an x and compute the gradient of the loss wrt the ws and bs. Then update ws and bs. Repeat. Epoch = one run through the training data.
def deep_learning(X, Y, learning_rate=1e-1, num_epochs=10, train_percent=80):
  num_examples = np.shape(X)[0]
  num_train = (num_examples*train_percent)/100
  X_train = X[:num_train,:,:]
  X_validation = X[num_train:,:,:]
  Y_train = Y[:num_train]
  Y_validation = Y[num_train:]
  print np.shape(X_train), np.shape(X_validation)
  num_hidden_1 = 50
  num_hidden_2 = 20
  num_output = 10

  x = np.random.randn(28,28)
  w1 = np.random.randn(num_hidden_1, 784)
  w2 = np.random.randn(num_hidden_2, num_hidden_1)
  w3 = np.random.randn(num_output, num_hidden_2)
  ws = [w1,w2,w3]
  b1 = np.random.randn(num_hidden_1)
  b2 = np.random.randn(num_hidden_2)
  b3 = np.random.randn(num_output)
  bs = [b1,b2,b3]

  for k in range(num_epochs):

    for i in range(num_train):
      out = forward(X_train[i,:,:], ws, bs)
      gws, gbs = dlossdws_and_bs(out, ws, bs, Y[i])

      for j in range(len(ws)):
        ws[j] = ws[j] - gws[j] * learning_rate
        bs[j] = bs[j] - gbs[j] * learning_rate
    # Output average loss after each epoch
    _, t_predictions, t_losses = get_outputs(X_train, Y_train, ws, bs)
    _, v_predictions, v_losses = get_outputs(X_validation, Y_validation, ws, bs)

    accuracy_t = accuracy(t_predictions, Y_train)
    accuracy_v = accuracy(v_predictions, Y_validation)

    print np.mean(t_losses), accuracy_t
    print np.mean(v_losses), accuracy_v, "validation"

  return ws, bs

# Do deep learning
#deep_learning(X, Y)

def read_mnist(filename):
  X = []
  Y = []
  f = open(filename, 'r')
  for line in f:
    # It's a csv file, i.e. comma separated value
    # Each line is '0,255,1,253'. If you do line.split(','), then you get [0,255,1,253] as a list. Then I just convert them to ints. The function map(a, b) applies the function a to each element in the list b. So map(int, [1.0,2.0]) = [1,2], where int is the function that converts things to integers.
    img = map(int, line.split(','))
    target = img[0]
    Y.append(target)
    img = np.array(img[1:])
    img = img.astype(float)
    # Normalise between 0 and 1.
    img /= 255.0
    # Reshape to get a 28x28 array
    img = np.reshape(img, (28,28))
    X.append(img)
  return np.array(X), np.array(Y)

# Loading test data
print "Reading test data"
X_test, Y_test = read_mnist('mnist_test.csv')
print np.shape(X_test), np.shape(Y_test)

# Loading train data
print "Reading training data"
X, Y = read_mnist('mnist_train.csv')
print np.shape(X), np.shape(Y)

# Doing deep learning
ws_trained, bs_trained = deep_learning(X,Y)

# Testing
_, test_predictions, _ = get_outputs(X_test, Y_test, ws_trained, bs_trained)

accuracy_test = accuracy(test_predictions, Y_test)
print "test accuracy is = ", accuracy_test
