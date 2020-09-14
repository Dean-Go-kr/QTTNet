# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:38:49 2019

@author: amax
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker
from matplotlib.patches import ConnectionPatch


################
###Core murge###
################
'''
## 1-2 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W1.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W2.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W3.npy')

## 2-1 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W4.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W5.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W6.npy')
core_4=np.load('/data/dong/modelnet/plot/core_before_mul/W7.npy')

## 2-2 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W8.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W9.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W10.npy')
core_4=np.load('/data/dong/modelnet/plot/core_before_mul/W11.npy')

## 2-3 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W12.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W13.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W14.npy')
core_4=np.load('/data/dong/modelnet/plot/core_before_mul/W15.npy')

## 3-1 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W16.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W17.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W18.npy')
core_4=np.load('/data/dong/modelnet/plot/core_before_mul/W19.npy')

## 3-2 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W20.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W21.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W22.npy')
core_4=np.load('/data/dong/modelnet/plot/core_before_mul/W23.npy')

## 3-3 ##
core_1=np.load('/data/dong/modelnet/plot/core_before_mul/W24.npy')
core_2=np.load('/data/dong/modelnet/plot/core_before_mul/W25.npy')
core_3=np.load('/data/dong/modelnet/plot/core_before_mul/W26.npy')
core_4=np.load('/data/dong/modelnet/plot/core_before_mul/W27.npy')
'''

'''
##LAYER 1-2##
core_1=np.load('/data/dong/modelnet/plot/W/W1.npy')
core_q_1=np.load('/data/dong/modelnet/plot/Wq/Wq1.npy')
core_2=np.load('/data/dong/modelnet/plot/W/W2.npy')
core_q_2=np.load('/data/dong/modelnet/plot/Wq/Wq2.npy')
core_3=np.load('/data/dong/modelnet/plot/W/W3.npy')
core_q_3=np.load('/data/dong/modelnet/plot/Wq/Wq3.npy')
core_4=np.load('/data/dong/modelnet/plot/W/W4.npy')
core_q_4=np.load('/data/dong/modelnet/plot/Wq/Wq4.npy')


##LAYER 2-1##
core_1=np.load('/data/dong/modelnet/plot/W/W5.npy')
core_q_1=np.load('/data/dong/modelnet/plot/Wq/Wq5.npy')
core_2=np.load('/data/dong/modelnet/plot/W/W6.npy')
core_q_2=np.load('/data/dong/modelnet/plot/Wq/Wq6.npy')
core_3=np.load('/data/dong/modelnet/plot/W/W7.npy')
core_q_3=np.load('/data/dong/modelnet/plot/Wq/Wq7.npy')
core_4=np.load('/data/dong/modelnet/plot/W/W8.npy')
core_q_4=np.load('/data/dong/modelnet/plot/Wq/Wq8.npy')
core_5=np.load('/data/dong/modelnet/plot/W/W9.npy')
core_q_5=np.load('/data/dong/modelnet/plot/Wq/Wq9.npy')

##LAYER 2-2##
core_1=np.load('/data/dong/modelnet/plot/W/W10.npy')
core_q_1=np.load('/data/dong/modelnet/plot/Wq/Wq10.npy')
core_2=np.load('/data/dong/modelnet/plot/W/W11.npy')
core_q_2=np.load('/data/dong/modelnet/plot/Wq/Wq11.npy')
core_3=np.load('/data/dong/modelnet/plot/W/W12.npy')
core_q_3=np.load('/data/dong/modelnet/plot/Wq/Wq12.npy')
core_4=np.load('/data/dong/modelnet/plot/W/W13.npy')
core_q_4=np.load('/data/dong/modelnet/plot/Wq/Wq13.npy')
core_5=np.load('/data/dong/modelnet/plot/W/W14.npy')
core_q_5=np.load('/data/dong/modelnet/plot/Wq/Wq14.npy')


##LAYER 2-3##
core_1=np.load('/data/dong/modelnet/plot/W/W15.npy')
#core_q_1=np.load('/data/dong/modelnet/plot/Wq/Wq15.npy')
core_2=np.load('/data/dong/modelnet/plot/W/W16.npy')
#core_q_2=np.load('/data/dong/modelnet/plot/Wq/Wq16.npy')
core_3=np.load('/data/dong/modelnet/plot/W/W17.npy')
#core_q_3=np.load('/data/dong/modelnet/plot/Wq/Wq17.npy')
core_4=np.load('/data/dong/modelnet/plot/W/W18.npy')
#core_q_4=np.load('/data/dong/modelnet/plot/Wq/Wq18.npy')
core_5=np.load('/data/dong/modelnet/plot/W/W19.npy')
#core_q_5=np.load('/data/dong/modelnet/plot/Wq/Wq19.npy')


##LAYER 3-1##
core_1=np.load('/data/dong/modelnet/correct_plot/W/W20.npy')
core_q_1=np.load('/data/dong/modelnet/correct_plot/Wq/Wq20.npy')
core_2=np.load('/data/dong/modelnet/correct_plot/W/W21.npy')
core_q_2=np.load('/data/dong/modelnet/correct_plot/Wq/Wq21.npy')
core_3=np.load('/data/dong/modelnet/correct_plot/W/W22.npy')
core_q_3=np.load('/data/dong/modelnet/correct_plot/Wq/Wq22.npy')
core_4=np.load('/data/dong/modelnet/correct_plot/W/W23.npy')
core_q_4=np.load('/data/dong/modelnet/correct_plot/Wq/Wq23.npy')
core_5=np.load('/data/dong/modelnet/correct_plot/W/W24.npy')
core_q_5=np.load('/data/dong/modelnet/correct_plot/Wq/Wq24.npy')


##LAYER 3-2##
core_1=np.load('/data/dong/modelnet/correct_plot/W/W25.npy')
core_q_1=np.load('/data/dong/modelnet/correct_plot/Wq/Wq25.npy')
core_2=np.load('/data/dong/modelnet/correct_plot/W/W26.npy')
core_q_2=np.load('/data/dong/modelnet/correct_plot/Wq/Wq26.npy')
core_3=np.load('/data/dong/modelnet/correct_plot/W/W27.npy')
core_q_3=np.load('/data/dong/modelnet/correct_plot/Wq/Wq27.npy')
core_4=np.load('/data/dong/modelnet/correct_plot/W/W28.npy')
core_q_4=np.load('/data/dong/modelnet/correct_plot/Wq/Wq28.npy')
core_5=np.load('/data/dong/modelnet/correct_plot/W/W29.npy')
core_q_5=np.load('/data/dong/modelnet/correct_plot/Wq/Wq29.npy')


##LAYER 3-3##
core_1=np.load('/data/dong/modelnet/correct_plot/W/W30.npy')
core_q_1=np.load('/data/dong/modelnet/correct_plot/Wq/Wq30.npy')
core_2=np.load('/data/dong/modelnet/correct_plot/W/W31.npy')
core_q_2=np.load('/data/dong/modelnet/correct_plot/Wq/Wq31.npy')
core_3=np.load('/data/dong/modelnet/correct_plot/W/W32.npy')
core_q_3=np.load('/data/dong/modelnet/correct_plot/Wq/Wq32.npy')
core_4=np.load('/data/dong/modelnet/correct_plot/W/W33.npy')
core_q_4=np.load('/data/dong/modelnet/correct_plot/Wq/Wq33.npy')
core_5=np.load('/data/dong/modelnet/correct_plot/W/W34.npy')
core_q_5=np.load('/data/dong/modelnet/correct_plot/Wq/Wq34.npy')
'''

'''
#length=len(BN.reshape(-1))
num_bin = 40
y_lim = 10
subplot = 4

def changey(temp,position):
    return round(temp/(length),4)

plt.figure(figsize=(20,5))
 
plt.subplots_adjust(wspace=0.3,hspace=0.2)
plt.subplot(1,subplot,1)
sns.kdeplot(core_1.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_1.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_1.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of 1st and 2nd TT-cores')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(1,subplot,2)
sns.kdeplot(core_2.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_2.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_2.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of 1st, 2nd, 3rd TT-cores')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)

plt.subplot(1,subplot,3)
sns.kdeplot(core_3.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_3.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_3.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of 1st, 2nd, 3rd, 4th TT-cores')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)

plt.subplot(1,subplot,4)
sns.kdeplot(core_4.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_4.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_4.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of all TT-cores')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)


plt.subplot(1,5,5)
sns.kdeplot(core_5.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_5.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_5.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 5th TT-core before quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)

plt.tight_layout()
plt.savefig('../fig/full_mul_3_3.pdf')
'''


###############
### Weight ####
###############

#plot: new running 
#renorm_full: full model / all_8
#w_4bit
#end4_center8
#all_4_0_003
#all_16_0_005
model = 'all_16_0_005'
layer = '3_3'

W_g1=np.load('/data/dong/modelnet/' + model +'/W/W30.npy')
W_g2=np.load('/data/dong/modelnet/' + model +'/W/W31.npy')
W_g3=np.load('/data/dong/modelnet/' + model +'/W/W32.npy')
W_g4=np.load('/data/dong/modelnet/' + model +'/W/W33.npy')
W_g5=np.load('/data/dong/modelnet/' + model +'/W/W34.npy')

Wq_g1=np.load('/data/dong/modelnet/' + model +'/Wq/Wq30.npy')
Wq_g2=np.load('/data/dong/modelnet/' + model +'/Wq/Wq31.npy')
Wq_g3=np.load('/data/dong/modelnet/' + model +'/Wq/Wq32.npy')
Wq_g4=np.load('/data/dong/modelnet/' + model +'/Wq/Wq33.npy')
Wq_g5=np.load('/data/dong/modelnet/' + model +'/Wq/Wq34.npy')



num_bin = 40
y_lim = 5
x_lim = -1
def changey(temp,position):
    return round(temp/(length),4)

plt.figure(figsize=(20,5))
 
plt.subplots_adjust(wspace=0.3,hspace=0.2)
plt.subplot(2,5,1)
sns.kdeplot(W_g1.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(W_g1.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W_g1.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 1st TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,2)
sns.kdeplot(W_g2.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(W_g2.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W_g2.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 2nd TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,3)
sns.kdeplot(W_g3.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(W_g3.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W_g3.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 3rd TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,4)
sns.kdeplot(W_g4.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(W_g4.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W_g4.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 4th TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,5)
sns.kdeplot(W_g5.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(W_g5.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W_g5.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 5th TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,6)
sns.distplot(Wq_g1.reshape(-1),color='m', bins = num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Wq_g1.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Wq_g1.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of quantized 1st TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,7)
sns.distplot(Wq_g2.reshape(-1),color='m', bins = num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Wq_g2.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Wq_g2.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of quantized 2nd TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,8)
sns.distplot(Wq_g3.reshape(-1),color='m', bins = num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Wq_g3.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Wq_g3.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of quantized 3rd TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,9)
sns.distplot(Wq_g4.reshape(-1),color='m', bins = num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Wq_g4.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Wq_g4.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of quantized 4th TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,10)
sns.distplot(Wq_g5.reshape(-1),color='m', bins = num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Wq_g5.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Wq_g5.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of quantized 5th TT-core')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.tight_layout()
plt.savefig('../fig/'+model+'_weight_'+layer+'.pdf')



#################
####MUL & ACTI###
#################

b_mul_g1=np.load('/data/dong/modelnet/' + model +'/mul_before/W24.npy')
b_mul_g2=np.load('/data/dong/modelnet/' + model +'/mul_before/W25.npy')
b_mul_g3=np.load('/data/dong/modelnet/' + model +'/mul_before/W26.npy')
b_mul_g4=np.load('/data/dong/modelnet/' + model +'/mul_before/W27.npy')

Acti_21=np.load('/data/dong/modelnet/' + model +'/A/A7.npy')

mul_g1=np.load('/data/dong/modelnet/' + model +'/mul/W24.npy')
mul_g2=np.load('/data/dong/modelnet/' + model +'/mul/W25.npy')
mul_g3=np.load('/data/dong/modelnet/' + model +'/mul/W26.npy')
mul_g4=np.load('/data/dong/modelnet/' + model +'/mul/W27.npy')

Acti_q_21=np.load('/data/dong/modelnet/' + model +'/Aq/Aq7.npy')

num_bin = 40
y_lim = 15
x_lim = -1
def changey(temp,position):
    return round(temp/(length),4)

plt.figure(figsize=(20,5))

plt.subplots_adjust(wspace=0.3,hspace=0.2)
plt.subplot(2,5,1)
sns.kdeplot(b_mul_g1.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(b_mul_g1.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(b_mul_g1.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of two TT-cores (31-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,2)
sns.kdeplot(b_mul_g2.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(b_mul_g2.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(b_mul_g2.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of three TT-cores (31-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,3)
sns.kdeplot(b_mul_g3.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(b_mul_g3.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(b_mul_g3.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of four TT-cores (31-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,4)
sns.kdeplot(b_mul_g4.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(b_mul_g4.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(b_mul_g4.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of all TT-cores (31-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,5)
sns.kdeplot(Acti_21.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,4)
plt.xlim(0,8)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Acti_21.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Acti_21.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of activation (31-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)
 
######
######

plt.subplots_adjust(wspace=0.3,hspace=0.2)
plt.subplot(2,5,6)
sns.distplot(mul_g1.reshape(-1),color='m', bins=num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(mul_g1.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(mul_g1.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of two TT-cores (16-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,7)
sns.distplot(mul_g2.reshape(-1),color='m', bins=num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(mul_g2.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(mul_g2.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of three TT-cores (16-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,8)
sns.distplot(mul_g3.reshape(-1),color='m' , bins=num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(mul_g3.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(mul_g3.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of all TT-cores (16-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.subplot(2,5,9)
sns.distplot(mul_g4.reshape(-1),color='m', bins=num_bin)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(x_lim,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(mul_g4.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(mul_g4.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Multiplication of all TT-cores (16-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,10)
sns.distplot(Acti_q_21.reshape(-1),color='m', bins=100)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,4)
plt.xlim(0,8)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Acti_q_21.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Acti_q_21.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of activation (16-bit)')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)

plt.tight_layout()
plt.savefig('../fig/'+model+'_mul_and_acti_'+layer+'.pdf')

'''
#############
#####QUAN####
#############
num_bin = 250
y_lim = 10

plt.subplots_adjust(wspace=0.3,hspace=0.2)
plt.subplot(2,5,6)
sns.distplot(core_q_1.reshape(-1),color='m', bins = num_bin, kde=True)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_q_1.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_q_1.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 1st TT-core after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,7)
sns.distplot(core_q_2.reshape(-1),color='m', bins = num_bin, kde=True)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_q_2.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_q_2.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 2nd TT-core after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)

plt.subplot(2,5,8)
sns.distplot(core_q_3.reshape(-1),color='m', bins = num_bin, kde=True)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_q_3.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_q_3.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 3rd TT-core after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)

plt.subplot(2,5,9)
sns.distplot(core_q_4.reshape(-1),color='m', bins = num_bin, kde=True, norm_hist=True)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_q_4.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_q_4.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 4th TT-core after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)

plt.subplot(2,5,10)
sns.distplot(core_q_5.reshape(-1),color='m', bins = num_bin, kde=False, norm_hist=True)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,y_lim)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(core_q_5.reshape(-1)),0,3,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(core_q_5.reshape(-1)),0,3,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of 5th TT-core after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)


plt.tight_layout()
plt.savefig('../fig/hist250_before_train_2_3__kde.pdf')


###########BN

#W8=np.load('/data/dong/modelnet/plot/BNq/BNq3.npy')
#WF=np.load('/data/dong/modelnet/plot/BN/BN3.npy')
#length=len(WF.reshape(-1))

def changey(temp,position):
    return round(temp/(length),10)
    
    
plt.subplot(2,2,1)
sns.kdeplot(BN.reshape(-1),color='m')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,12)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(BN.reshape(-1)),0,5,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(BN.reshape(-1)),0,5,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of BN before quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)


plt.subplot(2,2,3)
sns.kdeplot(BN_q.reshape(-1),color='m')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,12)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(BN_q.reshape(-1)),0,5,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(BN_q.reshape(-1)),0,5,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of BN after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)


plt.subplot(2,2,2)
sns.kdeplot(Acti.reshape(-1),color='m')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,12)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Acti.reshape(-1)),0,5,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Acti.reshape(-1)),0,5,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of Activation before quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)


plt.subplot(2,2,4)
sns.kdeplot(Acti_q.reshape(-1),color='m')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,12)
plt.xlim(-1,1)
plt.ylabel('Density')
#plt.xlabel('Expectation')
plt.vlines(np.max(Acti_q.reshape(-1)),0,5,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(Acti_q.reshape(-1)),0,5,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().set_title('Distribution of Activation after quantization')
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(c)',fontsize=8)


plt.tight_layout()
plt.savefig('../fig/BN_Acti.pdf')


W8=np.load('/data/dong/modelnet/plot/Aq/Aq3.npy')
WF=np.load('/data/dong/modelnet/plot/A/A3.npy')
length=len(WF.reshape(-1))

def changey(temp,position):
    return round(temp/(length),10)
    
    
plt.subplot(2,5,8)
sns.kdeplot(W8.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,1)
plt.ylabel('Density')
plt.xlabel('Expectation')
plt.vlines(np.max(W8.reshape(-1)),0,5,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W8.reshape(-1)),0,5,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,3)
sns.kdeplot(WF.reshape(-1),color='m')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
plt.ylim(0,1)
plt.ylabel('Density')
plt.xlabel('Expectation')
plt.vlines(np.max(WF.reshape(-1)),0,5,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(WF.reshape(-1)),0,5,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

###GGG
W8=np.load('/home/amax/yangyk/ExPerimentForPaperInception/COPY/ResNet18/WAGE/DIS8888/data/G/G5.npy')
WF=np.load('/home/amax/yangyk/ExPerimentForPaperInception/COPY/ResNet18/WAGE/DIS8888/data/Gq/Gq5.npy')
length=len(WF.reshape(-1))

def changey(temp,position):
    return round(temp/(length),10)
    
    
plt.subplot(2,5,9)
sns.kdeplot(W8.reshape(-1),color='red')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-2))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
#plt.ylim(0,1)
plt.xlim(-0.05,0.05)
plt.ylabel('Density')
plt.xlabel('Expectation')
plt.vlines(np.max(W8.reshape(-1)),0,50,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W8.reshape(-1)),0,50,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,4)
sns.kdeplot(WF.reshape(-1),color='m')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
#plt.ylim(0,1)
plt.xlim(-0.05,0.05)
plt.ylabel('Density')
plt.xlabel('Expectation')
plt.vlines(np.max(WF.reshape(-1)),0,50,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(WF.reshape(-1)),0,50,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#EEE
W8=np.load('/home/amax/yangyk/ExPerimentForPaperInception/COPY/ResNet18/WAGE/DIS8888/data/E2/E2_2.npy')
WF=np.load('/home/amax/yangyk/ExPerimentForPaperInception/COPY/ResNet18/WAGE/DIS8888/data/E2q/E2q2.npy')
length=len(WF.reshape(-1))

def changey(temp,position):
    return round(temp/(length),10)
    
    
plt.subplot(2,5,10)
sns.kdeplot(W8.reshape(-1),color='red')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
#plt.ylim(0,1)
plt.ylabel('Density')
plt.xlabel('Expectation')
plt.vlines(np.max(W8.reshape(-1)),0,50000,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(W8.reshape(-1)),0,50000,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.plot(sx,sy,"orange",linestyle='--',linewidth=1.5)
#plt.text(-2e-5,-50000,'(a)',fontsize=8)


plt.subplot(2,5,5)
sns.kdeplot(WF.reshape(-1),color='m')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5e-4))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(changey))
#plt.ylim(0,1)
plt.xlim(-0.001,0.001)
plt.ylabel('Density')
plt.xlabel('Expectation')
plt.vlines(np.max(WF.reshape(-1)),0,50000,linestyles='dashed',color='deeppink',label='upper bound')
plt.vlines(np.min(WF.reshape(-1)),0,50000,linestyles='dashed',color='green',label='lower bound')
plt.legend(fontsize=8) 
plt.grid(True) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

'''




#num=0
#maxW=np.max(W8.reshape(-1))
#for W in WF.reshape(-1):
#    if W>maxW/(2**7):
#        num=num+1
#print(num)
#print((1.0*num)/length)
        
    


#plt.figure(figsize=(16,8))
#
#x0=-2e-5
#x1=2e-5
#y0=10000
#y1=80000
#sx=[x0,x1,x1,x0,x0]
#sy=[y0,y0,y1,y1,y0]
#
#plt.subplot(121)
#sns.kdeplot(W8.reshape(-1),color='red')
#sns.kdeplot(W16.reshape(-1),color='darkgreen')
#sns.kdeplot(WF.reshape(-1),color='m')
#plt.axis([-2e-5,2e-5,10000,80000])









#gkde=stats.gaussian_kde(W8.reshape(-1))
#ind=np.linspace(-2e-5,2e-5,512)
#kdepdf=gkde.evaluate(ind)
#plt.plot(ind,kdepdf)
#
#W81=W8.reshape(-1)
#W81.plot(kind='kde')
#plt.show()
#
