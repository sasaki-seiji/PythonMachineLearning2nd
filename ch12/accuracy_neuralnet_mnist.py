import matplotlib.pyplot as plt

# 2019.10.27 add
import pickle
nn = pickle.load(open('neuralnet_mnist.pkl', 'rb'))

plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
#plt.savefig('images/12_08.png', dpi=300)
plt.show()
