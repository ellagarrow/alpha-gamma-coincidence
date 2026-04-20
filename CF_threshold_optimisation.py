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

    # Peak amplitude (negative)
    A = float(wf[pk])
    if A >= 0:
        return None  # not a negative pulse

    # Constant-fraction level is also negative
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

# when at least 0.03V to avoid lots of noise
thr_floor = 0.004 #V

# negative pulses
POLARITY_ALPHA = -1
POLARITY_GAMMA = -1

min_dist = 10          # samples between peaks
prom_a_V = 0.01      
prom_g_V = 0.01

max_dt_samp = 75 # 300 ns

cut_V = -0.9

# amplitude spectra things
bins = 300
a_range = (-2, 0)
g_range = (-2, 0)

a_hist = np.zeros(bins, dtype = np.int64)
g_hist = np.zeros(bins, dtype = np.int64)


# building coincidence pairs only, no fixed LE threshold here

dt_peak_ns = []          
a_cond_V = []            # coincident alpha amplitudes
g_cond_V = []            # coincident gamma amplitudes
coinc_examples = []
EXAMPLE_KEEP = 300

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


frac_A = 0.1
frac_G = 0.1


# In[9]:


frac_A = 0.1 # alpha fraction
frac_G = 0.1 # gamma fraction

# Recompute CF timing differences from stored coincidence pairs
def dt_from_pairs_CFD(pairs_use, frac_A, frac_G=None, search_back=80):

    if frac_G is None:
        frac_G = frac_A

    dt_list = []
    a_amp_list = []
    g_amp_list = []
    n_fail = 0 # where CF crossing failed

    for (i, a_pk, g_pk, a_amp, g_amp) in pairs_use:
        a_V = baseline_correct_to_volts(alpha_u16[i], baseline_samples)
        g_V = baseline_correct_to_volts(gamma_u16[i], baseline_samples)

        tA = cfd_time(a_V, a_pk, frac_A, dt_ns, search_back=search_back)
        tG = cfd_time(g_V, g_pk, frac_G, dt_ns, search_back=search_back)

        if (tA is None) or (tG is None):
            n_fail += 1
            continue

        dt_list.append(tG - tA)
        a_amp_list.append(a_amp)
        g_amp_list.append(g_amp)

    return (
        np.asarray(dt_list, dtype=float),
        np.asarray(a_amp_list, dtype=float),
        np.asarray(g_amp_list, dtype=float),
        n_fail
    )


# In[10]:


# function that fits a gaussian to prompt peak, similar to what is done for resolution later

def fit_prompt_gaussian(dt_ns, peak_lo=-10, peak_hi=25, peak_bins=200, tmin=-100, tmax=300):
    counts, edges = np.histogram(dt_ns, bins=peak_bins, range=(tmin, tmax))
    centres = 0.5 * (edges[:-1] + edges[1:])

    mask = (centres >= peak_lo) & (centres <= peak_hi)
    x = centres[mask]
    y = counts[mask]

    # define gaussian
    def gauss_bg(t, A, mu, sigma, C):
        return A * np.exp(-0.5 * ((t - mu) / sigma)**2) + C

    # initial guesses
    C0 = np.median(y)
    A0 = np.max(y) - C0
    mu0 = x[np.argmax(y)]
    sigma0 = 4.0

    p0 = [A0, mu0, sigma0, C0]
    sigma_y = np.sqrt(np.maximum(y, 1))

    popt, pcov = curve_fit(
        gauss_bg, x, y, p0=p0,
        sigma=sigma_y, absolute_sigma=True,
        maxfev=50000
    )

    perr = np.sqrt(np.diag(pcov))
    A, mu, sigma, C = popt
    A_err, mu_err, sigma_err, C_err = perr

    k = 2.0 * np.sqrt(2.0 * np.log(2.0))
    FWHM = k * abs(sigma)
    FWHM_err = k * sigma_err

    return {
        "counts": counts,
        "centres": centres,
        "popt": popt,
        "perr": perr,
        "mu": mu,
        "mu_err": mu_err,
        "sigma": sigma,
        "sigma_err": sigma_err,
        "FWHM": FWHM,
        "FWHM_err": FWHM_err,
        "peak_lo": peak_lo,
        "peak_hi": peak_hi,
        "tmin": tmin,
        "tmax": tmax,
    }

# returns vaLues
def prompt_fwhm_score(dt_arr, peak_lo=-10, peak_hi=25):
    try:
        fit = fit_prompt_gaussian(
            dt_arr,
            peak_lo=peak_lo,
            peak_hi=peak_hi,
            peak_bins=200,
            tmin=-100,
            tmax=300
        )
        return float(fit["mu"]), float(fit["FWHM"])
    except Exception:
        return np.nan, np.nan


# In[11]:

frac_grid = np.arange(0.02, 0.51, 0.01)   # scan 2% to 50%
scan_cfd = [] 

pairs_use = pairs  # limit for quicker test

# loops over fractions
for frac in frac_grid:
    dt_arr, a_amp_arr, g_amp_arr, n_fail = dt_from_pairs_CFD(
        pairs_use,
        frac_A=frac,
        frac_G=frac,
        search_back=80
    )

    n_ok = dt_arr.size

    if n_ok < 500:
        scan_cfd.append((frac, n_ok, n_fail, np.nan, np.nan))
        continue

    mu, fwhm = prompt_fwhm_score(dt_arr, peak_lo=-10, peak_hi=25)
    scan_cfd.append((frac, n_ok, n_fail, mu, fwhm))

scan_cfd = np.asarray(scan_cfd, dtype=float)

n_max = np.nanmax(scan_cfd[:, 1])
keep = scan_cfd[:, 1] >= 0.90 * n_max
cand = scan_cfd[keep]

# remove failed fits
cand = cand[np.isfinite(cand[:, 4])]

# finds best fraction
best = cand[np.argmin(cand[:, 4])]
best_frac, best_n, best_fail, best_mu, best_fwhm = best

# prints values
print("Best CFD fraction by prompt FWHM:")
print(f" frac = {best_frac:.3f}")
print(f" n_ok = {int(best_n)} / {int(n_max)}")
print(f" prompt mu = {best_mu:.3f} ns")
print(f" prompt FWHM = {best_fwhm:.3f} ns")


# In[20]:


# plots illustrating the above function

# threhsolds
plt.figure(figsize=(8,5))
plt.scatter(scan_cfd[:, 0], scan_cfd[:, 4], color='black', label='Data')
plt.axvline(best_frac, linestyle='--', color='r', label='Best CF fraction')
plt.xlabel("Fraction", fontsize = 20)
plt.ylabel("Prompt peak FWHM (ns)", fontsize = 20)
#plt.title("CFD fraction optimisation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend(fontsize = 15)
plt.savefig('CF_Threshold.png')
plt.show()

# number of events passing
plt.figure(figsize=(8,5))
plt.scatter(scan_cfd[:, 0], scan_cfd[:, 1], color='black')
plt.axvline(best_frac, linestyle='--', color='r', label='Best CFD fraction')
plt.xlabel("Constant fraction")
plt.ylabel("Accepted events")
plt.title("Accepted coincidences vs CFD fraction")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


# In[14]:


dt_cfd_best, a_cfd_best, g_cfd_best, n_fail_cfd_best = dt_from_pairs_CFD(
    pairs,
    frac_A=best_frac,
    frac_G=best_frac,
    search_back=80
)

print("Final CFD timing entries:", dt_cfd_best.size)
print("Failed crossings:", n_fail_cfd_best)


# In[ ]:




