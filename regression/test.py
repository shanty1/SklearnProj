from sklearn.metrics import r2_score
import numpy as np

if __name__ == '__main__':
    score = r2_score([1.7,1.2],[1.8,1.1])
    print(score)

    print(np.logspace(0, 3, 4))