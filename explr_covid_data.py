
# coding: utf-8

# In[2]:


import pandas as pd
import subprocess 

load_new = True #False

if load_new:
    print('Download data:')
    subprocess.call(['wget', '-O', 'data.csv', 'http://cowid.netlify.com/data/full_data.csv'])

data = pd.read_csv('data.csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d') #, data.dtypes

data.head()


# In[3]:


data.dtypes


# In[8]:


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime 
import numpy as np

def set_ticks(times, ax):
    tickevery = 4
    #ticks = list(map(lambda x: x.strftime("%d %b"), times))[::tickevery]
    
    ticks = np.datetime_as_string(times, unit='D')[::tickevery]
    
    ax.set_xticks(times[::tickevery])
    ax.set_xticklabels(ticks, rotation = 70, ha="right")
    
def plot_bar_datetime(times, y, ax, label = None, align = 'right', color = None):
    
    if align == 'right':
        times = times + np.timedelta64(12, 'h')
        
    ax.bar(times, y, width = .4, label = label, color = color) #, align = align)
        

def fit_exp(times, y, first_case_date):
    
    def exp_f_(times_, R, N):
        
        return np.exp(times_/R)*(N - y.cumsum())/N 
    
    times_ = (times - first_case_date - np.timedelta64(1, 'D'))/np.timedelta64(1, 'D')
    
    R, N = curve_fit(exp_f_, times_, y, [3, 1000])[0]
    
    model_newcases = exp_f_(times_, R, N)
    model_totcases = model_newcases.cumsum()
    return model_newcases, model_totcases, R, N
    
    
def plot_line_datetime(times, y, ax, label = None, color = None):
    
    ax.plot(times, y, label = label, c = color)
    

def plot_fig1(mask, title, ax):
    times = data.loc[mask, 'date'].values.astype('datetime64[D]')
    y_new = data.loc[mask, 'new_cases'].values
    y_tot = data.loc[mask, 'total_cases'].values

    ax2 = ax.twinx()
    y_new = np.nan_to_num(y_new)

    plot_bar_datetime(times, y_new, ax2, 'new_cases', align = 'right', color = 'C0')
    
    plot_bar_datetime(times, y_tot, ax, 'tot_cases', align = 'left', color = 'C1')
    
    try:
        y_fit_newcases, y_fit_totcases, R, N = fit_exp(times, y_new, times[y_new != 0][0])
    except RuntimeError:
        return
        
    label = r'fit: (1 - y/{:.0f})*e^[t[d]/{:.02f}], $\Delta$ T = {:.02f}d'.format(N, R, np.log(2)*R)
    plot_line_datetime(times, y_fit_newcases, ax2, label, color = 'C0')
    plot_line_datetime(times, y_fit_totcases, ax, 'fit: totcases', color = 'C1')
    
    if title is not None: ax.set_title(title)
    if label is not None: 
        ax2.legend(frameon = False, loc = 3)
        ax.legend(frameon = False, loc = 2)
    ax2.set_ylabel('total # cases')
    ax2.set_ylabel('# new cases')



# select countries:
minncases = 100
remove_countries = ['International', 'China']
countries = []
for country in data.location.unique():
    if not data.loc[data.loc[:, 'location'] == country, 'total_cases'].max() > minncases:
        continue
    if country in remove_countries:
        continue
    
    countries.append(country)


ncols = 2
nrows = len(countries)//ncols + 1

figW, figH = ncols * 8, nrows*4
fig, axarr = plt.subplots(nrows, ncols, figsize = (figW, figH), sharex = True)

for country, ax in zip(countries, axarr.ravel()):
    mask = data.location == country
    plot_fig1(mask, country, ax)

times = data.date.unique()
for i in range(len(axarr[0])): set_ticks(times, axarr[-1, i])
    

plt.tight_layout()
plt.savefig('countries.pdf', facecolor=fig.get_facecolor(), transparent=True)
plt.savefig('countries.png', dpi = 100, facecolor=fig.get_facecolor(), transparent=True)
plt.show()

