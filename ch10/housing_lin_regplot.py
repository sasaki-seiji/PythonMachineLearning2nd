import matplotlib.pyplot as plt

from lin_regplot import lin_regplot
from housing_lrgd import X_std, y_std, lr

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

#plt.savefig('images/10_06.png', dpi=300)
plt.show()
