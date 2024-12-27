import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../results/result.csv')

df_0 = df['0']
df_1 = df['1']
df_2 = df['2']
df_3 = df['3']




plt.plot(df_3)
# Add labels and title (optional)
plt.xlabel('Turn')
plt.ylabel('Accuracy')
plt.title('Learning-Unlearning plot')

# Show the plot
plt.show()
print('A')
#
# # # Save the figure
# plt.savefig('../plots/Learning-Unlearning.png')