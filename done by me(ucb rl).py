# -*- coding: utf-8 -*-
"""Copy of upper_confidence_bound.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g_qI20n5WS3TOf_UrktuEpw6Rnl5qsTB

# Upper Confidence Bound (UCB)

## Importing the libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""## Importing the dataset"""

ads=pd.read_csv('Ads_CTR_Optimisation.csv')

"""## Implementing UCB"""

import math
N = 10000
d = 10
no_of_selection=[0] * d
sum_of_rewards=[0]*d
ad_selected=[]
for i in range(0,N):
  ad=0
  max_bound=0
  for j in range(0,d):
    if (no_of_selection[j]>0):
      avg_reward=sum_of_rewards[j]/no_of_selection[j]
      deli=math.sqrt((3*(math.log(i+1)))/(2*(no_of_selection[j])))
      ucb=deli+avg_reward
    else:
      ucb=1e400
    if (ucb>max_bound):
      max_bound=ucb
      ad=j
  ad_selected.append(ad)
  no_of_selection[ad]+=1
  reward=ads.values[i,ad]
  sum_of_rewards[ad]=sum_of_rewards[ad]+reward

print(ad_selected)

"""## Visualising the results"""

plt.hist(ad_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()