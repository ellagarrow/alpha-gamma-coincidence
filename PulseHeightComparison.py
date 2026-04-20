#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size' : 10, "font.family" : "Times New Roman", "text.usetex" : True})


# ## Loading in Alpha data

# In[2]:


# load in alpha data from zenodo
url = " https://zenodo.org/records/18154472/files/wave1 (1).txt"

# these lines so can read in headerless files
r = requests.get(url)
print("HTTP status:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("First 200 chars:\n", r.text[:200])

prefix = "ch1_"
max_events = 100000 # larger than dataset 
samples = 1024 # samples per waveform 

response = requests.get(url, stream=True) # starts streaming file line by line, stream = True means dont load everything to memory at once
response.raise_for_status() # raises error if something fails

# setting up arrays for alpha data 
alpha = []
current_waveform = []
event_number = 0 # these will be used to store waveform samples as we go

for line in response.iter_lines(decode_unicode=True):
    if not line.strip(): #skips empty lines
        continue 

    value = float(line.strip())
    current_waveform.append(value)


    if len(current_waveform) == samples:
        alpha.append(current_waveform)
        event_number += 1
        current_waveform = []


        if max_events is not None and event_number >= max_events:
            break


print(f"Loaded {len(alpha)} waveforms.")
print(f"First waveform length: {len(alpha[0])} samples")


# ## Loading in gamma data

# In[3]:


# load in gamma data from zenodo
url1 = " https://zenodo.org/records/18154472/files/wave0 (1).txt"

# these lines so can read in headerless files
r = requests.get(url1)
print("HTTP status:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("First 200 chars:\n", r.text[:200])


prefix1 = "ch0_"
max_events = 100000
#header_lines = 7 # or None for all
samples1 = 1024

response1 = requests.get(url1, stream=True) # starts streaming file line by line, stream = True means dont load everything to memory at once
response1.raise_for_status() #raises error if something fails

gamma = []
current_waveform1 = []
#header_count1 = 0
event_number1 = 0 # these will be used to store waveform samples as we go

for line in response1.iter_lines(decode_unicode=True):
    if not line.strip(): #skips empty lines
        continue 

    value = float(line.strip())
    current_waveform1.append(value)


    if len(current_waveform1) == samples1:
        gamma.append(current_waveform1)
        event_number1 += 1
        current_waveform1 = []


        if max_events is not None and event_number1 >= max_events:
            break


print(f"Loaded {len(gamma)} waveforms.")
print(f"First waveform length: {len(gamma[0])} samples")


# Dataframe 

# In[4]:


df_gamma = pd.DataFrame(gamma).transpose()
df_gamma.columns = [f"waveform_{i}" for i in range(len(df_gamma.columns))]
df_gamma.head()


# In[5]:


df_alpha = pd.DataFrame(alpha).transpose()
df_alpha.columns = [f"waveform_{i}" for i in range(len(df_alpha.columns))]
df_alpha.head()


# ## Correcting data and identifying peaks

# In[6]:


# defining the time axis
dt = 1 / 250e6 # = 4ns per sample. 
time = np.arange(1024) * dt #converts numbers 1-1024 to 4ns steps

# conversion from ADC to voltage
N_bits = 12
V_scale = 1
ADC_max = 2**(N_bits-1)

#converting ADC counts to volts
gamma_waveform = df_gamma * (V_scale / ADC_max)
alpha_waveform = df_alpha * (V_scale / ADC_max)


# defining baseline function using start and end of each waveform
def baseline(wf, n=100):
    start_mean = np.mean(wf[:n])
    end_mean = np.mean(wf[-n:])
    return max(start_mean, end_mean) 

# setting up for the corrected (baseline subtracted) waveforms and peaks
alpha_correct = {}
gamma_correct = {}

alpha_peaks = {}
gamma_peaks = {}

for col in alpha_waveform.columns:     # looping through each waveform
                                                                       
    wf_a = alpha_waveform[col].to_numpy()     # converts from panda series to numpy array 
    wf_g = gamma_waveform[col].to_numpy()

    # baseline subtraction/correction
    wf_a_corrected = wf_a - baseline(wf_a)                                
    wf_g_corrected = wf_g - baseline(wf_g)

    alpha_correct[col] = wf_a_corrected       # saving the corrected waveforms in dictionaries, keyed by waveform no. 
    gamma_correct[col] = wf_g_corrected

    # Find all negative peaks, invert waveform with negative sign 
    peaks_a, _ = find_peaks(-wf_a_corrected, prominence=0.01, distance = 10)    # requiring a pulse amplitude of at least 0.2V relative to baseline
    peaks_g, _ = find_peaks(-wf_g_corrected, prominence=0.01, distance = 10)    # hard coded discriminator 

    alpha_peaks[col] = peaks_a      # saves all detected pulses for each waveform
    gamma_peaks[col] = peaks_g



# ## Amplitude identification

# In[7]:


# taking amplitudes by reading waveform value at peak index

alpha_amplitudes = []          # arrays where amplitudes will be stored
gamma_amplitudes = []

for col in alpha_correct:
    if len(alpha_peaks[col]) > 0:   # skips waveforms where 0 pulses detected 
        pk = alpha_peaks[col]  # selects all pulses
        alpha_amplitudes.extend(alpha_correct[col][pk])   # appends the pulse amplitude to array

    if len(gamma_peaks[col]) > 0:
        pk = gamma_peaks[col]
        gamma_amplitudes.extend(gamma_correct[col][pk])


# In[8]:


#plotting the amplitude histogram

plt.figure(figsize=(10,5))
plt.hist(alpha_amplitudes, bins=500)
plt.xlabel("Pulse amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title('Alpha Pulse Amplitude Spectrum', fontsize = 20)
plt.yscale('log')
plt.show()

plt.figure(figsize=(10,5))
plt.hist(gamma_amplitudes, bins=500)
plt.xlabel("Pulse amplitude (V)",fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title('Gamma Pulse Amplitude Spectrum', fontsize = 20)
plt.yscale('log')
plt.show()


# ## Trigger Condition Spectrum

# ### Gamma Amplitude: Condition that Alpha detector fired and the gamma is the biggest within 300ns after

# In[9]:


a_amplitude_index = {} 
g_amplitude_index = {}

for col in alpha_correct:
    a_pk = alpha_peaks[col]
    g_pk = gamma_peaks[col]

    a_amplitude_index[col] = alpha_correct[col][a_pk] if len(a_pk) else np.array([])  # finds waveforms where there is alpha pulse
    g_amplitude_index[col] = gamma_correct[col][g_pk] if len(g_pk) else np.array([])  # keeps and stores waveforms where there isnt too


# In[10]:


a_cond_spectrum = []
g_cond_spectrum = []
dt_coinc_ns = []
coincidence_events = []

thr_a = 0.05  # units (V) or ADC counts
thr_g = 0.05

max_dt = 75

for col in alpha_correct.keys():

    a_idx = np.array(alpha_peaks[col])     # gets peak positions in samples
    g_idx = np.array(gamma_peaks[col])

    if a_idx.size == 0 or g_idx.size == 0:        # skips where no pulses
        continue
    
    a_amp = alpha_correct[col][a_idx]
    g_amp = gamma_correct[col][g_idx]

    a_mask = a_amp < -thr_a
    g_mask = g_amp < -thr_g
    a_idx, a_amp = a_idx[a_mask], a_amp[a_mask]   # keeping those that pass conditions
    g_idx, g_amp = g_idx[g_mask], g_amp[g_mask]

    if len(a_idx) == 0 or len(g_idx) == 0:
        continue

    for a_i, a_a in zip(a_idx, a_amp):
        cand = np.where(g_idx <= a_i + max_dt)[0]
    
        if cand.size == 0:
            continue 

        j = cand[np.argmax(np.abs(g_amp[cand]))]
        g_cond_spectrum.append(g_amp[j])
        dt_coinc_ns.append((g_idx[j] - a_i) * dt * 1e9)

    for g_i, g_a in zip(g_idx, g_amp):
        cand = np.where(a_idx <= g_i + max_dt)[0]
    
        if cand.size == 0:
            continue 

        l = cand[np.argmax(np.abs(a_amp[cand]))]
        a_cond_spectrum.append(a_amp[l])
        dt_coinc_ns.append((a_idx[l] - g_i) * dt * 1e9)


    for a_i in a_idx:
        cand = np.where((g_idx > a_i) & (g_idx <= a_i + max_dt))[0]
        
        if cand.size == 0:
            continue

        j = cand[np.argmax(np.abs(g_amp[cand]))]          # biggest candidate (position)
        g_i = g_idx[j]       # sample number

        t_alpha = a_i * dt
        t_gamma = g_i * dt
        coincidence_events.append((col, t_alpha, t_gamma, (g_i - a_i) * dt * 1e9))


# In[11]:


plt.figure(figsize=(10,5))
plt.hist(g_cond_spectrum, bins=500)
plt.xlabel("Pulse amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title('Gamma Pulse Amplitude Spectrum, Condition that Alpha fired', fontsize = 20)
plt.yscale('log')
plt.show()

plt.figure(figsize=(10,5))
plt.hist(a_cond_spectrum, bins=500)
plt.xlabel("Pulse amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title('Alpha Pulse Amplitude Spectrum, Condition that Gamma fired', fontsize = 20)
plt.yscale('log')
plt.show()


# In[12]:


print("N(alpha, gamma fired) = ", len(a_cond_spectrum))
print("N(gamma, alpha fired) = ", len(g_cond_spectrum))


# ## Coincidence events finding

# ### NOTE: want to amend this to use every coincident event not just the first in each

# ## Printing and plotting some waveforms to see the timings of the coincident events

# In[13]:


for event in coincidence_events[15:25]:
    wf, ta, tg, dt_ns = event
    print(f"Waveform {wf}:  alpha at {ta*1e9:.1f} ns, gamma at {tg*1e9:.1f} ns, delta_t = {dt_ns:.1f} ns")


# In[14]:


for col, ta, tg, dt_ns in coincidence_events[15:25]:
    plt.figure(figsize=(10,5))
    
    plt.plot(time, alpha_correct[col], label="alpha")
    plt.plot(time, gamma_correct[col], label="gamma")
    
    plt.axvline(ta, color='blue', linestyle='--')
    plt.axvline(tg, color='orange', linestyle='--')

    plt.xlabel('Time (s)', fontsize = 20)
    plt.ylabel('Voltage (V)', fontsize = 20)
    plt.title(f"Coincidence in waveform {col}",fontsize = 20)
    plt.legend(fontsize = 15)
    plt.show()


# In[15]:


t_alpha_list = []
t_gamma_list = []
delta_t_list = []   # t_gamma - t_alpha

for col, t_alpha, t_gamma, dt in coincidence_events:
    t_alpha_list.append(t_alpha)
    t_gamma_list.append(t_gamma)
    delta_t_list.append(t_gamma - t_alpha)

t_alpha_ns = np.array(t_alpha_list) * 1e9
t_gamma_ns = np.array(t_gamma_list) * 1e9
delta_t_ns = np.array(delta_t_list) * 1e9


# In[16]:


# plt.figure(figsize=(7,4))
# plt.hist(delta_t_ns, bins=200)
# plt.xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
# plt.ylabel("Counts", fontsize = 20)
# plt.title("Timing spectrum for alpha - gamma coincidences", fontsize = 20)
# plt.show()


# In[17]:


counts, bin_edges = np.histogram(delta_t_ns, bins = 200)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])


# In[18]:


mask = (bin_centres > 20) & (counts > 0)
t_fit = bin_centres[mask]
N_fit = counts[mask]
sigma_n = np.sqrt(N_fit)


# In[19]:


def exp_decay(t, N0, tau, B):
    return N0 * np.exp(-t / tau) + B

p0 = [N_fit[0], 100.0, 0.0] #initial guess

popt, pcov = curve_fit(exp_decay, t_fit, N_fit, p0=p0, sigma=sigma_n, absolute_sigma=True)

N0_fit, tau_fit, B_fit = popt
tau_err = np.sqrt(np.diag(pcov))[1]

print(f"Lifetime τ = {tau_fit:.1f} ± {tau_err:.1f} ns")


# In[20]:


plt.figure(figsize=(8,4))
plt.hist(delta_t_ns, bins = 200, label="Data")

t_plot = np.linspace(20, max(delta_t_ns), 500)
plt.plot(t_plot, exp_decay(t_plot, *popt), 'r-', 
         label=f"Fit (tau = {tau_fit:.1f} ± {tau_err:.1f} ns)")

plt.xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.legend(fontsize = 15)
plt.title("Timing spectrum with exponential fit", fontsize = 20)
plt.show()


# ## 2D Pulse height correlation

# In[21]:


a_1 = []
g_1 = []

for evt in alpha_correct.keys():
    
    a_idx = np.asarray(alpha_peaks[evt])
    g_idx = np.asarray(gamma_peaks[evt])
    
    if a_idx.size == 0 or g_idx.size == 0:
        continue 

    a_amp = alpha_correct[evt][a_idx]
    g_amp = gamma_correct[evt][g_idx]

    for ai, aa in zip(a_idx, a_amp):
        
        if aa > -thr_a:
            continue

        # find gamma for this alpha
        
        cand = np.where((g_idx > ai) & (g_idx <= ai + max_dt))[0]
        
        if cand.size == 0:
            continue

        j = cand[np.argmax(np.abs(g_amp[cand]))]
        a_1.append(aa)
        g_1.append(g_amp[j])


# In[22]:


plt.figure(figsize=(7,6))
plt.hist2d(a_1, g_1, bins=[200, 200], cmap = 'plasma')
plt.colorbar(label="Counts")
plt.xlabel("Alpha pulse height (V)")
plt.ylabel("Gamma pulse height (V)")
plt.title("Pulse height correlation")
plt.show()


# ## Time between alpha peaks

# In[23]:


dt_alpha_alpha_ns = []

for col in alpha_peaks.keys():
    a_idx = np.array(alpha_peaks[col])

    # need at least two alpha peaks to form a difference
    if a_idx.size < 2:
        continue

    # ensure peaks are time-ordered
    a_idx = np.sort(a_idx)

    # consecutive differences in samples
    da = np.diff(a_idx)

    # convert to ns
    da_ns = da * dt

    dt_alpha_alpha_ns.extend(da_ns)


# In[24]:


plt.figure(figsize=(7,5))
plt.hist(dt_alpha_alpha_ns, bins=200, histtype='step')
plt.yscale('log')
plt.xlabel(r'$\Delta t_{\alpha\alpha}$ (ns)')
plt.ylabel('Counts')
plt.title('Time separation between consecutive alpha pulses')
plt.grid(alpha=0.3)
plt.show()


# In[ ]:




