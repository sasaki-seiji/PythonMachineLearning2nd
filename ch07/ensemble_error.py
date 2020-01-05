from scipy.special import comb
import math

def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)




# 2020.01.05 change
#ensemble_error(n_classifier=11, error=0.25)
err = ensemble_error(n_classifier=11, error=0.25)
print('ensemble error: ', err)