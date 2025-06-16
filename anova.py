import numpy as np
from scipy.stats import f_oneway, tukey_hsd

cnn_predictive_perf = np.array([52.616215, 48.14901, 50.355663, 52.9264, 46.734562, 52.266586, 51.50814, 45.268517, 49.721214, 48.850174, 50.902733, 55.247448, 47.502907, 53.66998, 51.75543, 58.526165, 47.658478, 50.8758, 53.71434, 53.60551, 53.43616, 48.43492, 51.023354, 47.320087, 55.471653, 55.855076, 51.108227])
lstm_predictive_perf = np.array([55.999336, 62.60312, 65.209625, 43.34095, 50.91176, 45.40563, 57.459515, 52.954056, 65.1629, 63.095673, 51.049267, 53.35548, 52.667274, 54.79724, 48.29207, 51.369728, 42.9839, 50.938873, 50.4173, 43.538826, 52.461773, 50.32902, 62.42275, 46.838425, 54.65701, 68.95676, 56.365917])
cnnlstm_predictive_perf = np.array([48.58929, 49.37242, 47.82046, 62.18865, 46.32206, 56.69337, 53.68413, 46.705223, 53.623833, 47.17823, 46.376602, 55.761433, 47.612343, 48.044384, 55.083656, 60.453217, 47.56868, 55.17548, 56.308598, 47.853012, 50.044254, 45.5239, 48.526413, 51.37785, 65.47981, 53.592857, 49.51165])

f_statistics, p_value = f_oneway(cnn_predictive_perf, lstm_predictive_perf, cnnlstm_predictive_perf)

print(f"f-Statistic: {f_statistics}\np-Value: {p_value}\n")

alpha = 0.05
if p_value < alpha:
    print(f"Reject the null hypothesis. There is significant difference between the predictive performances.")

    # If there is significant difference, perform Tukey's HSD
    print(f"{tukey_hsd(cnn_predictive_perf, lstm_predictive_perf, cnnlstm_predictive_perf)}")
else:
    print(f"Fail to reject the null hypothesis. There is no significant difference between the predictive performances.")