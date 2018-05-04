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
    print np.shape(q)
    dloss_dws.insert(0, np.dot(q, p))
    dloss_dbs.insert(0, dloss_du_i)
    dloss_dx_i = np.dot(np.transpose(ws[k]),dloss_du_i)
    dloss_du_i = dloss_dx_i*x*(1-x)
    layer -= 2
  return dloss_dws, dloss_dbs


# Then do numerical gradient check. i.e. check that dL_dw1_ij = (L(w1_ij+eps) - #L(w1_ij-eps)) / (2*eps).
# u1 = w1*x1 + b1
# x2 = sigmoid(u1)
# u2 = w2*x2 + b2
# y = softmax(u2)

# Matrix multiplication: np.dot(A,B)
# Matrix transpose: np.transpose(B)

 # Temporarily, let w = ws[-1], b = bs[-1].
 # u = ws[-1] * x + bs[-1]
 # Next, we want dlossdw_ij and dlossdb_i
 # Want to calculate it in terms of dlossdu_i and du_idw_jk

# place for changing things
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

print loss(out[-1],t)

print len(out)

for i in range(len(out)):
  print np.shape(out[i])

gws, gbs = dlossdws_and_bs(out, ws, bs, t)

for i in range(len(gws)):
  print np.shape(gws[i])

e = 1e-5

def test_grad(x, ws, bs):

  gwts = []

  for k in range(len(ws)):
    gwt = np.zeros(np.shape(ws[k]))

    for i in range(np.shape(gwt)[0]):

      for j in range(np.shape(gwt)[1]):
        wc = np.copy(ws[k])
        wc[i,j] += e
        wsc = np.copy(ws)
        wsc[k] = wc
        loss_plus = loss(forward(x, wsc, bs)[-1], t)
        wc[i,j] -= 2*e
        loss_minus = loss(forward(x, wsc, bs)[-1], t)
        gwt[i,j] = (loss_plus - loss_minus)/(2*e)

    gwts.append(gwt)

  return gwts

gwts = test_grad(x, ws, bs)

for i in range(len(ws)):

  comp = gws[i] - gwts[i]

  print np.max(np.abs(comp))

  print np.max(np.abs(comp)/(e + np.abs(gws[i]) + np.abs(gwts[i])))
