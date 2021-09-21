import numpy as np

def softmax(a): #소프트맥스 출력 : 0~1, 출력 총합 : 1
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y