import numpy as np
from properscoring import crps_ensemble

# 读取 y_true 和 y_pred 数据
y_true = np.load('y_true_batch_5.npy')
y_pred = np.load('y_pred_batch_5.npy')

# 获取数据的维度
B, T_p, V, D = y_true.shape
n_samples = y_pred.shape[1]

# 初始化 CRPS 结果数组
crps_results = np.zeros((V, D, T_p))

# 计算每个节点的每个通道在 T_p 长度内的 CRPS
for v in range(V):
    for d in range(D):
        for t in range(T_p):
            crps_results[v, d, t] = crps_ensemble(y_true[:, t, v, d], y_pred[:, :, t, v, d])

mean_crps = np.mean(crps_results, axis=2)

# 打印或保存 CRPS 结果
print("CRPS Results:", crps_results)
np.save('crps_results_1.npy', crps_results)