#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import numpy as np
import os, platform 
import string
import matplotlib.pyplot as plt
import requests
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size' : 10, "font.family" : "Times New Roman", "text.usetex" : True})


# In[2]:


print(os.listdir("/home/2660162g/Masters Project/BinFiles"))


# In[3]:


RAW_DIR = "/home/2660162g/Masters Project/BinFiles"

GAMMA_RAW = os.path.join(RAW_DIR, "Gammalongfeb_u16.bin")
ALPHA_RAW = os.path.join(RAW_DIR, "Alphalongfeb_u16.bin")

record_length = 256
dtype = np.uint16


def load_waveforms_memmap(bin_path, record_length=256, dtype=np.uint16):
    
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")

    print("Loading:", bin_path)

    data = np.memmap(bin_path, dtype=dtype, mode="r")

    n_waveforms = data.size // record_length
    n_use = n_waveforms * record_length

    pulses = data[:n_use].reshape(n_waveforms, record_length)

    return data, pulses, n_waveforms



gamma_data, gamma_u16, Ng = load_waveforms_memmap(GAMMA_RAW, record_length)
alpha_data, alpha_u16, Na = load_waveforms_memmap(ALPHA_RAW, record_length)

N = min(Ng, Na)  # safe if one file is slightly shorter
print("Gamma shape:", gamma_u16.shape)
print("Alpha shape:", alpha_u16.shape)
print("Using N waveforms:", N)


# In[4]:


# timing axis conversion
fs = 250e6 # sampling rate
dt = 1.0 / fs # timing interval
dt_ns = dt * 1e9 # nanosecond conversion
t_ns = np.arange(record_length) * dt * 1e9 # full axis nanosecond conversion

# ADC to Volts
Vpp = 2.0 # voltage range
ADC_levels = 4096  # 12-bit
V_per_count = Vpp / ADC_levels # voltage conversion


# In[5]:


# baseline correction
baseline_samples = 50 # can be altered 

def baseline_correct_to_volts(wf_u16: np.ndarray,
                              baseline_samples: int = 50) -> np.ndarray:
   
    wf = wf_u16.astype(np.int32, copy=False)
    b = np.median(wf[baseline_samples:]).astype(np.float32)  # using only start
    corr_counts = wf.astype(np.float32) - b # correcting the counts
    return corr_counts * V_per_count  # volts


# In[6]:


def cfd_time(wf: np.ndarray, pk: int, frac: float, dt: float,
             search_back: int = 80,
             return_samples: bool = False):
   
    wf = np.asarray(wf)
    n = wf.size
    if n < 2:
        return None

    pk = int(pk)
    if pk <= 0 or pk >= n:
        return None

    A = float(wf[pk])
    if A >= 0:
        return None  # not a negative pulse

    # CF level is also negative
    level = frac * A  

    lo = max(1, pk - int(search_back))
    hi = min(pk, n - 1)

    for k in range(hi, lo, -1):
        y0 = wf[k - 1]
        y1 = wf[k]
        if (y0 > level) and (y1 <= level):
            denom = (y1 - y0)
            if denom == 0:
                t_samp = float(k)
            else:
                frac_lin = (level - y0) / denom  # between 0 and 1
                t_samp = (k - 1) + float(frac_lin)
            return t_samp if return_samples else t_samp * dt

    return None


# In[7]:


# Thresholds
# fraction of maximum pulse height
thr_frac = 0.1

# avoid lots of noise
thr_floor = 0.004 #V

# negative pulses
POLARITY_ALPHA = -1
POLARITY_GAMMA = -1

min_dist = 10          # samples between peaks
prom_a_V = 0.01      
prom_g_V = 0.01

cut_V = -0.7 

max_dt_samp = 75 # 300 ns

# amplitude spectra things
bins = 300
a_range = (-2, 0)
g_range = (-2, 0)

a_hist = np.zeros(bins, dtype = np.int64)
g_hist = np.zeros(bins, dtype = np.int64)

dt_peak_ns = []         
a_cond_V = []            # coincident alpha amplitudes
g_cond_V = []            # coincident gamma amplitudes
coinc_examples = []
EXAMPLE_KEEP = 30

pairs = []               # store pairs

processed = 0
skipped_a = 0
skipped_g = 0

for i in range(N):
    a_V = baseline_correct_to_volts(alpha_u16[i], baseline_samples)
    g_V = baseline_correct_to_volts(gamma_u16[i], baseline_samples)

    # make pulses positive-going for peak finding
    if POLARITY_ALPHA == -1:
        a_sig = -a_V
    else:
        a_sig = a_V

    if POLARITY_GAMMA == -1:
        g_sig = -g_V
    else:
        g_sig = g_V

    # dynamic pulse-finding thresholds
    a_max = float(np.max(a_sig))
    thr_a_this = max(thr_floor, thr_frac * a_max)

    if a_max < thr_floor:
        a_idx = np.array([], dtype=np.int32)
        skipped_a += 1
    else:
        a_idx, _ = find_peaks(
            a_sig,
            height=thr_a_this,
            prominence=prom_a_V,
            distance=min_dist
        )

    g_max = float(np.max(g_sig))
    thr_g_this = max(thr_floor, thr_frac * g_max)

    if g_max < thr_floor:
        g_idx = np.array([], dtype=np.int32)
        skipped_g += 1
    else:
        g_idx, _ = find_peaks(
            g_sig,
            height=thr_g_this,
            prominence=prom_g_V,
            distance=min_dist
        )

    # all-pulse spectra
    if a_idx.size:
        h, _ = np.histogram(a_V[a_idx], bins=bins, range=a_range)
        a_hist += h

    if g_idx.size:
        g_idx = g_idx[g_V[g_idx] >= cut_V]
        h, _ = np.histogram(g_V[g_idx], bins=bins, range=g_range)
        g_hist += h

    # coincidence matching: store peak pairs only
    if a_idx.size and g_idx.size:
        a_amp = a_V[a_idx]
        g_amp = g_V[g_idx]

        for a_i, a_a in zip(a_idx, a_amp):
            lo = a_i - max_dt_samp
            hi = a_i + max_dt_samp

            cand = np.where((g_idx >= lo) & (g_idx <= hi))[0]
            if cand.size == 0:
                continue

            # choose gamma peak closest in time to this alpha peak
            j = cand[np.argmin(np.abs(g_idx[cand] - a_i))]
            g_i = int(g_idx[j])
            g_a = float(g_amp[j])

            pairs.append((i, int(a_i), int(g_i), float(a_a), g_a))
            a_cond_V.append(float(a_a))
            g_cond_V.append(g_a)

            if len(coinc_examples) < EXAMPLE_KEEP:
                coinc_examples.append((i, int(a_i), int(g_i), float(a_a), g_a))

    processed += 1
    if processed % 100_000 == 0:
        print(f"{processed:,}/{N:,} | skipped_a={skipped_a:,} skipped_g={skipped_g:,} | pairs={len(pairs):,}")

a_cond_V = np.asarray(a_cond_V, dtype=np.float32)
g_cond_V = np.asarray(g_cond_V, dtype=np.float32)

print("Done.")
print("Stored pairs:", len(pairs))
print("Stored examples:", len(coinc_examples))


# In[8]:


frac_A = 0.1 # alpha fraction
frac_G = 0.1 # gamma fraction


# In[9]:


# computes leading edge and CF times for each pair

def timing_from_pairs_LE_CFD(pairs_use, thr_A, frac_A, thr_G, frac_G, search_back=80):
   
    if thr_G is None:
        thr_G = thr_A

    dt_LE = []
    dt_CFD = []
    
    a_amp_list = []
    g_amp_list = []

    tA_LE_list = []
    tG_LE_list = []
    tA_CFD_list = []
    tG_CFD_list = []
    
    n_fail = 0

    for (i, a_pk, g_pk, a_amp, g_amp) in pairs_use:
        a_V = baseline_correct_to_volts(alpha_u16[i], baseline_samples)
        g_V = baseline_correct_to_volts(gamma_u16[i], baseline_samples)

        tA_LE = leading_edge_time(a_V, a_pk, thr_A, dt_ns, search_back=search_back) # le times
        tG_LE = leading_edge_time(g_V, g_pk, thr_G, dt_ns, search_back=search_back)

        tA_CFD = cfd_time(a_V, a_pk, frac_A, dt_ns, search_back=search_back) # cf times
        tG_CFD = cfd_time(g_V, g_pk, frac_G, dt_ns, search_back=search_back)

        if (tA_LE is None) or (tG_LE is None) or (tA_CFD is None) or (tG_CFD is None):
            n_fail += 1
            continue

        # coincidence timing
        dt_LE.append(tG_LE - tA_LE)
        dt_CFD.append(tG_CFD - tA_CFD)

        # store individual times
        tA_LE_list.append(tA_LE)
        tG_LE_list.append(tG_LE)
        tA_CFD_list.append(tA_CFD)
        tG_CFD_list.append(tG_CFD)

        a_amp_list.append(a_amp)
        g_amp_list.append(g_amp)

    return (
        np.asarray(dt_LE),
        np.asarray(dt_CFD),
        np.asarray(a_amp_list),
        np.asarray(g_amp_list),
        np.asarray(tA_LE_list),
        np.asarray(tG_LE_list),
        np.asarray(tA_CFD_list),
        np.asarray(tG_CFD_list),
        n_fail
    )


# In[13]:


# defing what the two walks are

alpha_walk = tA_LE - tA_CFD
gamma_walk = tG_LE - tG_CFD


# In[14]:


from matplotlib.colors import PowerNorm # for the plot


# In[26]:


# plotting the walks

plt.figure(figsize=(8,6))

ticks = [1, 10, 100, 200, 400, 600, 800] # for colour scale

# alpha walk
h1 = plt.hist2d(
    a_amp_arr,
    alpha_walk,
    bins=[120, 120],
    range=[[-1.81, -0.001], [-5, 5]], 
    norm=PowerNorm(gamma=0.5)
)

cbar1 = plt.colorbar(h1[3], ticks = ticks)   # attach colorbar to histogram

cbar1.set_label("Counts", fontsize=15)      # label size
cbar1.ax.tick_params(labelsize=12)          # tick label size
plt.xlabel("Alpha pulse amplitude (V)", fontsize = 20)
plt.ylabel(r"$t_{LE,\alpha}-t_{CFD,\alpha}$ (ns)", fontsize = 20)
plt.title("Alpha time walk relative to CFD", fontsize = 20)
plt.savefig('TimeWalkAlpha.png')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

# gamma walk
h2 = plt.hist2d(
    g_amp_arr,
    gamma_walk,
    bins=[120,120],
    range=[[-0.7,-0.001],[-5,5]],
    norm=PowerNorm(gamma=0.5)
)

cbar2 = plt.colorbar(h2[3], ticks = ticks)   # attach colorbar to histogram

cbar2.set_label("Counts", fontsize=15)      # label size
cbar2.ax.tick_params(labelsize=12)          # tick label size
plt.xlabel("Gamma pulse amplitude (V)", fontsize = 20)
plt.ylabel(r"$t_{LE,\gamma}-t_{CFD,\gamma}$ (ns)", fontsize = 20)
plt.title("Gamma time walk relative to CFD", fontsize = 20)
plt.savefig('TimeWalkGamma.png')
plt.tight_layout()
plt.show()


# In[ ]:


# started trying to fix and got confused


# In[16]:


def mean_in_bins(x, y, bins, min_count=30):
    idx = np.digitize(x, bins)
    xc, ym = [], []

    for k in range(1, len(bins)):
        mask = idx == k
        if np.sum(mask) >= min_count:
            xc.append(0.5 * (bins[k-1] + bins[k]))
            ym.append(np.mean(y[mask]))

    return np.asarray(xc), np.asarray(ym)


# In[17]:


bins = np.linspace(np.min(a_amp_arr), np.max(a_amp_arr), 40)

xc, ym = mean_in_bins(a_amp_arr, alpha_walk, bins)

plt.figure(figsize=(8,6))

plt.hist2d(a_amp_arr, alpha_walk, bins=[120,120], range=[[-1.8,-0.001],[-5,5]])

plt.plot(xc, ym, 'r-', lw=2)

plt.xlabel("Alpha pulse amplitude (V)")
plt.ylabel(r"$t_{LE}-t_{CFD}$ (ns)")
plt.title("Leading-edge time walk")

plt.colorbar(label="Counts")
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(8,6))
plt.hist2d(
    a_amp_arr[train], alpha_walk[train],
    bins=[120,120],
    range=[[-1.8, -0.001], [-5, 2]]
)
plt.plot(xA_corr, yA_corr, 'r-', lw=2, label='Median correction')
plt.colorbar(label="Counts")
plt.xlabel("Alpha pulse amplitude (V)")
plt.ylabel(r"$t_{LE,\alpha} - t_{CFD,\alpha}$ (ns)")
plt.title("Alpha time-walk correction curve")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist2d(
    g_amp_arr[train], gamma_walk[train],
    bins=[120,120],
    range=[[-1.8, -0.001], [-5, 2]]
)
plt.plot(xG_corr, yG_corr, 'r-', lw=2, label='Median correction')
plt.colorbar(label="Counts")
plt.xlabel("Gamma pulse amplitude (V)")
plt.ylabel(r"$t_{LE,\gamma} - t_{CFD,\gamma}$ (ns)")
plt.title("Gamma time-walk correction curve")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




