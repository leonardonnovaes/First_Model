import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 8)
def get_linear_curve(x, w, b=0, noise_scale=0):
  return w * x + b + noise_scale * np.random.randn(x.shape[0])
x = np.arange(-10, 30.1, 0.5)
Y = get_linear_curve(x, 1.8, 32, noise_scale = 2.5)
x.shape, Y.shape
plt.scatter(x, Y)
plt.xlabel('°C', fontsize=20)
plt.ylabel('°F', fontsize=20)
#Inicializar
w = np.random.rand(1)
b =0
def forward(inputs, w, b):
  return w * inputs + b
def mse(Y, y):
  return (Y-y)**2
def backpropagation(inputs, outputs, targets, w, b, lr):
  dw = lr*(-2*inputs*(targets-outputs)).mean()
  db = lr*(-2*(targets-outputs)).mean()
  w -= dw
  b -= db
  return w, b
def model_fit(inputs, target, w, b, epochs= 200, lr = 0.001):
  for epoch in range(epochs):
    outputs= forward(inputs, w, b)
    loss = np.mean(mse(target, outputs))
    w, b = backpropagation(inputs, outputs, target, w, b, lr)

    if(epoch+1) % (epochs/10) ==0:
      print(f'Epoch: [{(epoch+1)}/{epochs}] Loss: [{loss:.4f}]')
  return w, b
x = np.arange(-10,30,2)
Y = get_linear_curve(x, w = 1.8, b=32)
#Inicialização

w = np.random.randn(1)
b = np.zeros(1)

w, b = model_fit(x, Y, w, b, epochs=2000, lr=0.002)
print(f'W: {w[0]:.3f}, b: {b[0]:.3f}')