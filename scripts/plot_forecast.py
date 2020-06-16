import numpy as np
import csv
import matplotlib.pyplot as plt

with open('../Results/Seattle_Test_Results.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all3 = [row for row in reader]
data_all3=np.array(data_all3, dtype=float)

with open('../Data_Processed_New/Seattle_data_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all_sea = [row for row in reader]
data_all_sea = np.array(data_all_sea, dtype=float)


error_1=(data_all3[-14*24:, 1]-data_all_sea[-14*24:, 0])/data_all_sea[-14*24:, 0]
error_2=(data_all3[-14*24:, 2]-data_all_sea[-14*24:, 0])/data_all_sea[-14*24:, 0]
error_3=(data_all3[-14*24:, 3]-data_all_sea[-14*24:, 0])/data_all_sea[-14*24:, 0]
error_4=(data_all3[-14*24:, 4]-data_all_sea[-14*24:, 0])/data_all_sea[-14*24:, 0]
print(error_1)
print("Here", np.mean(error_1))
fig, ax = plt.subplots(1, 4, figsize=(21,5))
ax[0].hist(error_1, bins=10, edgecolor='black',linestyle=('dashed'), color='orange', weights=np.ones(len(error_1)) / len(error_1))
ax[0].set_ylabel('Probability')
ax[0].set_xlabel('Error Deviation Ratio')
ax[0].grid(axis='y')
ax[0].set_xlim([-0.5, 0.5])
ax[0].title.set_text('NN_Orig')
ax[1].hist(error_2, bins=10, edgecolor='black',linestyle=('dashed'), color='olive', weights=np.ones(len(error_1)) / len(error_1))
ax[1].set_xlabel('Error Deviation Ratio')
ax[1].grid(axis='y')
ax[1].set_xlim([-0.5, 0.5])
ax[1].title.set_text('Retrain')
ax[2].hist(error_3, bins=10, edgecolor='black',linestyle=('dashed'),color='deepskyblue',  weights=np.ones(len(error_1)) / len(error_1))
ax[2].set_xlabel('Error Deviation Ratio')
ax[2].grid(axis='y')
ax[2].set_xlim([-0.5, 0.5])
ax[2].title.set_text('Mobi')
ax[3].hist(error_4, bins=10, edgecolor='black',linestyle=('dashed'), color='limegreen', weights=np.ones(len(error_1)) / len(error_1))
ax[3].set_xlabel('Error Deviation Ratio')
ax[3].grid(axis='y')
ax[3].set_xlim([-0.5, 0.5])
ax[3].title.set_text('Mobi_MTL')


plt.show()


print(np.shape(data_all3))
ax = plt.subplot(1, 1, 1)
ax.plot(data_all3[-14*24:, 1], 'orange',label='NN_Orig', lw=2)
ax.plot(data_all3[-14*24:, 2], 'olive',label='Retrain',lw=2)
ax.plot(data_all3[-14*24:, 3], 'deepskyblue',label='Mobi',lw=2)
ax.plot(data_all3[-14*24:, 4], 'limegreen',label='Mobi_MTL',lw=2)
ax.plot(data_all_sea[-14*24:, 0], 'crimson',lw=2.5, label='Actual Load', linestyle='-.')
ax.set_xlim([0, 24*14])
ax.set_ylabel('Load (MW)')
ax.legend()
labels = ['May 2', 'May 6', 'May 10', 'May 14']
x = np.arange(len(labels))/0.0104  # the label locations
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(linestyle='--')
plt.show()