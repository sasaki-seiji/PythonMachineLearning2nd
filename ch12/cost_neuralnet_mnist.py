import matplotlib.pyplot as plt

# 2019.10.27 add
import pickle
nn = pickle.load(open('neuralnet_mnist.pkl', 'rb'))

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('images/12_07.png', dpi=300)
plt.show()

