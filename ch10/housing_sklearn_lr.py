import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from lin_regplot import lin_regplot
from housing_pd import df

# 2019.10.13 add
X = df[['RM']].values
y = df['MEDV'].values

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

#plt.savefig('images/10_07.png', dpi=300)
plt.show()
