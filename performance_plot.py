# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:12:13 2017

@author: zmx
"""
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_GROUPS = 6
scores = (0.98,0.87,0.86,0.82,0.96,0.93)
index = np.arange(NUMBER_OF_GROUPS)
bar_width = 0.6
opacity = 0.7
results = plt.bar(index - bar_width/2, scores, bar_width, alpha = opacity)
plt.xlabel("Methodology")
plt.ylabel("Precision")
plt.title("Performance of different methodologies")
plt.xticks(index, ("SVM","KNN","RF","NB","ANN","LR"))
#plt.legend()
plt.tight_layout()
plt.show()