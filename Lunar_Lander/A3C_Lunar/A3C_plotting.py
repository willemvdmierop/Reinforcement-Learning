import pandas as pd
import matplotlib.pyplot as plt


A3C_ON = True
A3C_ON_40 = True
A3C_lr_001 = True
A3C_lr_001_40 = True
path_scores = "/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Optimization/Lunar_latest_code_27st_april/A3C/A3C_scores/"

######################## scores plot #########################
df = pd.read_csv(path_scores + "scores_A3C_Test_episodes.csv")
df2 = pd.read_csv(path_scores + "scores_A3C_Test_episodes_40steps.csv")
df3 = pd.read_csv(path_scores + "scores_A3C_Test_episodes_lr_001.csv")
df4 = pd.read_csv(path_scores + "scores_A3C_Test_episodes_lr_001_40.csv")



fig, ax = plt.subplots(1,2,figsize = (18,8))
window = 50
print("last value A3C 20",df.iloc[-1])
print("last value A3C 40",df2.iloc[-1])
print("last value A3C lr = 0.001 forward steps 20",df3.iloc[-1])
print("last value A3C lr = 0.001 forward steps 40",df4.iloc[-1])
rolling_mean = pd.Series(df['Scores']).rolling(window).mean()
std = pd.Series(df['Scores']).rolling(window).std()
rolling_mean2 = pd.Series(df2['Scores']).rolling(window).mean()
std2 = pd.Series(df2['Scores']).rolling(window).std()
rolling_mean3 = pd.Series(df3['Scores']).rolling(window).mean()
std3 = pd.Series(df3['Scores']).rolling(window).std()
rolling_mean4 = pd.Series(df4['Scores']).rolling(window).mean()
std4 = pd.Series(df4['Scores']).rolling(window).std()
if A3C_ON:
    ax[0].plot(rolling_mean, label = 'A3C 20 forward steps, lr = 0.0001', color = 'blue')
    ax[0].fill_between(range(len(pd.Series(df['Scores']))), rolling_mean - std, rolling_mean + std, color = 'blue', alpha = 0.1)

if A3C_ON_40:
    ax[0].plot(rolling_mean2, label = 'A3C 40 forward steps, lr = 0.0001', color = 'yellow')
    ax[0].fill_between(range(len(pd.Series(df2['Scores']))), rolling_mean2 - std2, rolling_mean2 + std2, color='yellow',
                       alpha=0.1)

if A3C_lr_001:
    ax[0].plot(rolling_mean3, label = 'A3C 20 forward steps, lr = 0.001', color = 'green')
    ax[0].fill_between(range(len(pd.Series(df3['Scores']))), rolling_mean3 - std3, rolling_mean3 + std3, color = 'green', alpha = 0.1)

if A3C_lr_001_40:
    ax[0].plot(rolling_mean4, label = 'A3C 40 forward steps, lr = 0.001', color = 'red')
    ax[0].fill_between(range(len(pd.Series(df4['Scores']))), rolling_mean4 - std4, rolling_mean4 + std4, color='red',
                       alpha=0.1)

ax[0].set_title('Scores moving average ({}-episode window)'.format(window))
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Score")
ax[0].legend(loc = 'lower right')


######################## Episode length plot ###############
rolling_mean_length = pd.Series(df['Episode length']).rolling(window).mean()
std_length = pd.Series(df['Episode length']).rolling(window).std()
rolling_mean_length2 = pd.Series(df2['Episode length']).rolling(window).mean()
std_length2 = pd.Series(df2['Episode length']).rolling(window).std()
rolling_mean_length3 = pd.Series(df3['Episode length']).rolling(window).mean()
std_length3 = pd.Series(df3['Episode length']).rolling(window).std()
rolling_mean_length4 = pd.Series(df4['Episode length']).rolling(window).mean()
std_length4 = pd.Series(df4['Episode length']).rolling(window).std()

if A3C_ON:
    ax[1].plot(rolling_mean_length, label = 'A3C 20 forward steps, lr = 0.0001', color = 'blue')
    ax[1].fill_between(range(len(pd.Series(df['Scores']))), rolling_mean_length - std_length, rolling_mean_length + std_length, color = 'blue', alpha = 0.1)

if A3C_ON_40:
    ax[1].plot(rolling_mean_length2, label = 'A3C 40 forward steps, lr = 0.0001', color = 'yellow')
    ax[1].fill_between(range(len(pd.Series(df2['Scores']))), rolling_mean_length2 - std_length2, rolling_mean_length2 + std_length2, color='yellow',
                       alpha=0.1)

if A3C_lr_001:
    ax[1].plot(rolling_mean_length3, label = 'A3C 20 forward steps, lr = 0.001', color = 'green')
    ax[1].fill_between(range(len(pd.Series(df3['Scores']))), rolling_mean_length3 - std_length3, rolling_mean_length3 + std_length3, color = 'green', alpha = 0.1)

if A3C_lr_001_40:
    ax[1].plot(rolling_mean_length4, label = 'A3C 40 forward steps, lr = 0.001', color = 'red')
    ax[1].fill_between(range(len(pd.Series(df4['Scores']))), rolling_mean_length4 - std_length4, rolling_mean_length4 + std_length4, color='red',
                       alpha=0.1)


ax[1].set_title('Episode Length moving average ({}-episode window)'.format(window))
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Episode Length")
ax[1].legend(loc = 'lower right')
fig.subplots_adjust(hspace=1)
plt.show()
