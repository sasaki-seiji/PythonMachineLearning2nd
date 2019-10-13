import matplotlib.pyplot as plt
import seaborn as sns

from housing_pd import df

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

# 2019.10.13 change
#sns.pairplot(df[cols], size=2.5)
#plt.tight_layout()
## plt.savefig('images/10_03.png', dpi=300)
#plt.show()
if __name__ == '__main__':
    sns.pairplot(df[cols], size=2.5)
    plt.tight_layout()
    # plt.savefig('images/10_03.png', dpi=300)
    plt.show()
