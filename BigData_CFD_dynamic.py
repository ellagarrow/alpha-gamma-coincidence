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


# ## Loading in Alpha data

# ## Loading in gamma data

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


# timing 
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
baseline_samples = 50

def baseline_correct_to_volts(wf_u16: np.ndarray,
                              baseline_samples: int = 50) -> np.ndarray:
   
    wf = wf_u16.astype(np.int32, copy=False)
    b = np.median(wf[:baseline_samples]).astype(np.float32)  # using only start 
    corr_counts = wf.astype(np.float32) - b # correcting the counts
    return corr_counts * V_per_count  # volts


# In[6]:


# constant fraction

def cfd_time(wf: np.ndarray, pk: int, frac: float, dt: float,
             search_back: int = 80,
             return_samples: bool = False):
   
    wf = np.asarray(wf)
    n = wf.size
    if n < 2:  # need at least two samples to have a crossing
        return None

    pk = int(pk)
    if pk <= 0 or pk >= n:
        return None

    A = float(wf[pk])
    if A >= 0:
        return None  # not a negative pulse

    # constant fraction level is also negative
    level = frac * A   

    lo = max(1, pk - int(search_back))
    hi = min(pk, n - 1)

    # interpolation

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
# fraction of maximum pulse height, for dynamic pulse finding
thr_frac = 0.1 # 10%

# when at least 0.03V to avoid lots of noise
thr_floor = 0.004 #V

# negative pulses
POLARITY_ALPHA = -1
POLARITY_GAMMA = -1

cut_V = -0.7

min_dist = 10          # samples between peaks
prom_a_V = 0.01      
prom_g_V = 0.01

#min_dt_samp = 0
max_dt_samp = 75 # 300 ns coincidence window

# amplitude spectra things
bins = 300
a_range = (-2, 0)
g_range = (-2, 0)

a_hist = np.zeros(bins, dtype = np.int64)
g_hist = np.zeros(bins, dtype = np.int64)


dt_peak_ns = []               # timing spectrum values
a_cond_V = []                 # alpha amps in coincidence sepctrum
g_cond_V = []                 # gamma amps in coincidence spectrum
coinc_examples = []           # store a few examples for plotting later 

EXAMPLE_KEEP = 30


processed = 0
skipped_a = 0    # seeing how many dont have waveforms
skipped_g = 0

for i in range(N):
    a_V = baseline_correct_to_volts(alpha_u16[i], baseline_samples)  # the baseline correction
    g_V = baseline_correct_to_volts(gamma_u16[i], baseline_samples)


    if POLARITY_ALPHA == -1:
        a_sig = -a_V  # positive-going for find_peaks

        # per-waveform dynamic threshold = 10% of max pulse height
        a_max = float(np.max(a_sig))                # same as -a_V.min()
        thr_a_this = max(thr_floor, thr_frac * a_max) # keeps whatever is larger 

        if a_max < thr_floor:   # skips pulses below floor level
            a_idx = np.array([], dtype=np.int32)
            skipped_a += 1
        else:
            a_idx, _ = find_peaks(a_sig, height=thr_a_this,
                              prominence=prom_a_V, distance=min_dist)


    if POLARITY_GAMMA == -1: # repeat for gamma
        g_sig = -g_V

        g_max = float(np.max(g_sig))
        thr_g_this = max(thr_floor, thr_frac * g_max)

        if g_max < thr_floor:
            g_idx = np.array([], dtype=np.int32)
            skipped_g += 1
        else:
            g_idx, _ = find_peaks(g_sig, height=thr_g_this,
                              prominence=prom_g_V, distance=min_dist)


    if a_idx.size:
        h, _ = np.histogram(a_V[a_idx], bins=bins, range=a_range)    # histograms all pulses 
        a_hist += h
    if g_idx.size:
        g_idx = g_idx[g_V[g_idx] >= cut_V]
        h, _ = np.histogram(g_V[g_idx], bins=bins, range=g_range)
        g_hist += h


    if a_idx.size and g_idx.size:
        a_amp = a_V[a_idx]
        g_amp = g_V[g_idx]


         # coincidence part 
        for a_i in a_idx:
            lo = a_i - max_dt_samp
            hi = a_i + max_dt_samp

            cand = np.where((g_idx >= lo) & (g_idx <= hi))[0]
            if cand.size == 0:
                continue

            # choose gamma peak closest in time to this alpha peak
            j = cand[np.argmin(np.abs(g_idx[cand] - a_i))]
            g_i = int(g_idx[j])

            # choose a constant fraction
            CFD_FRAC_A = 0.1
            CFD_FRAC_G = 0.1

            tA_ns = cfd_time(a_V, int(a_i), CFD_FRAC_A, dt_ns, search_back=80)
            tG_ns = cfd_time(g_V, int(g_i), CFD_FRAC_G, dt_ns, search_back=80)

            if (tA_ns is None) or (tG_ns is None):
                continue

            dt_ns_val = tG_ns - tA_ns # coincidence times
            dt_peak_ns.append(dt_ns_val)
                
            a_cond_V.append(float(a_V[a_i]))
            g_cond_V.append(float(g_amp[j]))

            if len(coinc_examples) < EXAMPLE_KEEP:
                coinc_examples.append((i, int(a_i), int(g_i), float(tA_ns), float(tG_ns)))

    # progress checker
    processed += 1
    if processed % 100_000 == 0:
        print(f"{processed:,}/{N:,} | skipped_a={skipped_a:,} skipped_g={skipped_g:,} | dt entries={len(dt_peak_ns):,}")

dt_peak_ns = np.array(dt_peak_ns, dtype=np.float64)
a_cond_V = np.array(a_cond_V, dtype=np.float32)
g_cond_V = np.array(g_cond_V, dtype=np.float32)

print("Done.")
print("dt_peak_ns entries:", dt_peak_ns.size)
print("Stored examples:", len(coinc_examples))


# In[8]:


# amplitude spectra (all peaks) 
bins_a = a_hist.size
edges_a = np.linspace(a_range[0], a_range[1], bins_a + 1)
cent_a  = 0.5 * (edges_a[:-1] + edges_a[1:])
width_a = edges_a[1] - edges_a[0]
nz = np.where(a_hist > 0)[0]
idx0_a = nz[0] if nz.size else 0
x_start_a = edges_a[max(0, idx0_a - 5)]

bins_g = g_hist.size
edges_g = np.linspace(g_range[0], g_range[1], bins_g + 1)
cent_g  = 0.5 * (edges_g[:-1] + edges_g[1:])
width_g = edges_g[1] - edges_g[0]
nz = np.where(g_hist > 0)[0]
idx0_g = nz[0] if nz.size else 0
x_start_g = edges_g[max(0, idx0_g - 5)]


plt.figure(figsize=(9,6))
plt.bar(cent_a, a_hist, width=width_a, align="center", color = 'black')
plt.yscale("log")
plt.xlim(x_start_a, a_range[1])
plt.xlabel("Alpha peak amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Alpha amplitude spectrum (all peaks)", fontsize = 20)
plt.savefig('AlphaALL_CFD.png')
plt.show()

plt.figure(figsize=(9,6))
plt.bar(cent_g, g_hist, width=width_g, align="center", color = 'black')
plt.yscale("log")
plt.xlim(x_start_g, g_range[1])
plt.xlabel("Gamma peak amplitude (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Gamma amplitude spectrum (all peaks)", fontsize = 20)
#plt.savefig('GammaALL_CFD.png')
plt.show()






# coincident amplitude spectra 
plt.figure(figsize=(9,6))
plt.hist(g_cond_V, bins=300, color = 'black')
plt.yscale("log")
plt.xlabel("Gamma amplitude in coincidences (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Gamma amplitude spectrum, used in coincidences", fontsize = 20)
#plt.savefig('GammaCoincidence_CFD.png')
plt.show()

plt.figure(figsize=(9,6))
plt.hist(a_cond_V, bins=300, color = 'black')
plt.yscale("log")
plt.xlabel("Alpha amplitude in coincidences (V)", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.title("Alpha amplitude spectrum, used in coincidences", fontsize = 20)
#plt.savefig('AlphaCoincidence_CFD.png')
plt.show()


# In[15]:


# making timing spectrum

delta_t_ns = np.asarray(dt_peak_ns, dtype=float)
delta_t_ns = delta_t_ns[np.isfinite(delta_t_ns)]
#delta_t_ns = delta_t_ns[delta_t_ns >= 0]

t_max_plot = max_dt_samp * dt_ns
delta_t_ns = delta_t_ns[delta_t_ns <= t_max_plot] # coincidence window

# histogram set up
bins_t = 150
t_min_plot = 0.0

counts, edges = np.histogram(delta_t_ns, bins=bins_t, range=(-t_max_plot, t_max_plot))
centres = 0.5*(edges[:-1] + edges[1:])

lo_t_cut = 15  # avoiding prompt peak
hi_t_cut = 300
mask = (centres >= lo_t_cut) & (centres <= hi_t_cut) & (counts > 0)

t_fit = centres[mask]
N_fit = counts[mask]

# defining exponential
def exp_plus_bg(t, N0, tau, B):
    return N0*np.exp(-t/tau) + B

# Initial guesses
B_guess = float(np.median(N_fit[-10:])) if len(N_fit) >= 10 else 0.0    # background from end of exponential
N0_guess = max(N_fit[0] - B_guess, 1.0)       
tau_guess = 67.0    # lifetime guess 
p0 = [N0_guess, tau_guess, B_guess]

# error, poisson 
sigma = np.sqrt(N_fit) # for fit
sigma1 = np.sqrt(counts) # for error bars

popt, pcov = curve_fit(exp_plus_bg, t_fit, N_fit, p0=p0, sigma=sigma, absolute_sigma=True,maxfev=50_000)
perr = np.sqrt(np.diag(pcov))
N0, tau, B = popt
dN0, dtau, dB = perr

# fit
t_smooth = np.linspace(lo_t_cut, hi_t_cut, 800)
fit_smooth = exp_plus_bg(t_smooth, *popt)

# residuals and normalised residuals 
model_fit = exp_plus_bg(t_fit, *popt)   # predicted counts
residuals = N_fit - model_fit           # actual counts
norm_resid = residuals / np.sqrt(np.maximum(model_fit, 1))

nr_mean = float(np.mean(norm_resid))
nr_std  = float(np.std(norm_resid, ddof=1)) if norm_resid.size > 1 else float("nan")

# number of points
N = norm_resid.size

# mean
nr_mean = float(np.mean(norm_resid)) if N > 0 else float("nan")

# standard deviation
nr_std = float(np.std(norm_resid, ddof=1)) if N > 1 else float("nan")

# error on mean 
nr_mean_err = nr_std / np.sqrt(N) if N > 1 else float("nan")

# error on standard deviation
nr_std_err = nr_std / np.sqrt(2*(N - 1)) if N > 1 else float("nan")

# print nicely
print(f"Normalised residuals:")
print(f"  mean = {nr_mean:.4f} ± {nr_mean_err:.4f}")
print(f"  std  = {nr_std:.4f} ± {nr_std_err:.4f}")

print(f"Lifetime τ = {tau:.2f} ± {dtau:.2f} ns")
#print(f"Normalised residuals mean = {nr_mean:.3f}, std = {nr_std:.3f}")


fig = plt.figure(figsize=(9, 11))
gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.2, 1.2, 1.8], hspace=0.45)

# figures
# 1: timing spectrum + fit
ax1 = fig.add_subplot(gs[0, 0])
ax1.grid(alpha = 0.3, zorder = 0)
ax1.hist(delta_t_ns, bins=bins_t, range=(-t_max_plot, t_max_plot), alpha=0.4, label="Data", color = 'black', zorder = 2)
ax1.plot(t_smooth, fit_smooth, linewidth=2, label=rf"Fit ($\tau ={tau:.2f}\pm{dtau:.2f}$ ns)", color = 'r')
ax1.errorbar(centres, counts, yerr=sigma1, fmt='none',
             ecolor='black', elinewidth=1, capsize=2, zorder=3)
#ax1.set_title("Timing spectrum with exponential fit", fontsize = 20)
ax1.set_ylabel("Counts", fontsize = 20)
ax1.set_xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
ax1.set_xlim(t_min_plot, t_max_plot)
ax1.legend(fontsize = 15)

# 2: residuals vs t 
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.errorbar(t_fit, residuals, yerr=sigma, fmt='o', capsize=2, color = 'black')
ax2.axhline(0, linestyle='--', linewidth=1, color = 'red')
ax2.set_title("Constant Fraction Timing, Residuals", fontsize = 20)
ax2.set_ylabel("Residuals", fontsize = 20)
ax2.set_xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
ax2.grid(alpha= 0.3, zorder = 0)
ax2.set_xlim(t_min_plot, t_max_plot)

# 3: normalised residual distribution
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(norm_resid, bins=15, density=True, alpha=0.8, color = 'black', zorder = 2)

x = np.linspace(-5, 5, 400)
gauss01 = (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)
ax3.plot(x, gauss01, linestyle='--', linewidth=2, label="N(0,1)", color = 'r')
ax3.axvline(0, linestyle=':', linewidth=1, color= 'r')

#ax3.set_title("Normalised residuals distribution", fontsize = 20)
ax3.set_xlabel(r"Normalised residual", fontsize = 20)
ax3.set_ylabel("Probability density", fontsize = 20)
ax3.grid(alpha= 0.3, zorder = 0)
ax3.legend(fontsize = 15)

plt.savefig('Res_CF.png')
plt.show()


# In[10]:


# printing some examples for visual checks
if len(coinc_examples) == 0:
    print("No coincidence examples stored (coinc_examples is empty).")
else:
    def baseline_correct_to_volts_local(wf_u16):
        wf = wf_u16.astype(np.int32, copy=False)
        b = np.median(wf[:baseline_samples]).astype(np.float32)
        return (wf.astype(np.float32) - b) * V_per_count

    for (i, a_pk, g_pk, tA_ns, tG_ns) in coinc_examples[29:30]:
        a_V = baseline_correct_to_volts_local(alpha_u16[i])
        g_V = baseline_correct_to_volts_local(gamma_u16[i])

        plt.figure(figsize=(10,5))
        plt.plot(t_ns, a_V, label="Alpha", color = 'black')
        plt.plot(t_ns, g_V, label="Gamma", color = 'red')
        plt.axvline(tA_ns, linestyle="--", label="Alpha CF time", color = 'black')
        plt.axvline(tG_ns, linestyle="--", label="Gamma CF time", color = 'red')
        plt.plot(tA_ns, -0.02, 'x', markersize=10, label="Threshold", color = 'b')
        plt.plot(tG_ns, -0.01, 'x', markersize=10, color = 'b')
        #plt.xlim(250, 400)
        plt.xlabel("Time (ns)", fontsize = 20)
        plt.ylabel("Voltage (V)", fontsize = 20)
        plt.title(f"Constant Fraction Example",fontsize = 20)
        plt.grid(alpha= 0.3)
        plt.legend(fontsize = 15)
        plt.savefig('CFExample.png')
        plt.show()


# In[11]:


# peak fitting for resolution

peak_bins = 200
tmin, tmax = -100, 300
counts, edges = np.histogram(delta_t_ns, bins=peak_bins, range=(tmin, tmax))
centres = 0.5 * (edges[:-1] + edges[1:])


# prompt peak fit window
peak_lo, peak_hi = -10, 25

peak_mask = (centres >= peak_lo) & (centres <= peak_hi)
x = centres[peak_mask]
y = counts[peak_mask]

# define gaussian (+ constant maybe)
def gauss_bg(t, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((t - mu) / sigma)**2) + C

# initial guesses from the window
C0 = np.median(y)  # baseline guess
A0 = np.max(y) - C0  # amplitude guess
mu0 = x[np.argmax(y)]  # mean guess
sigma0 = 4 # sigma guess

p0 = [A0, mu0, sigma0, C0]

sigma_y = np.sqrt(np.maximum(y, 1))
popt, pcov = curve_fit(gauss_bg, x, y, p0=p0, sigma=sigma_y, absolute_sigma=True)

A, mu, sigma, C = popt
perr = np.sqrt(np.diag(pcov))
A_err, mu_err, sigma_err, C_err = perr

# FWHM 
k = 2.0 * np.sqrt(2.0 * np.log(2.0))      # 2.35482...
FWHM = k * sigma
FWHM_err = k * sigma_err

print(f"Prompt Gaussian fit:")
print(f"  mu    = {mu:.3f} ± {mu_err:.3f} ns")
print(f"  sigma = {sigma:.3f} ± {sigma_err:.3f} ns")
print(f"  FWHM  = {FWHM:.3f} ± {FWHM_err:.3f} ns")



# plotting
t_fit = np.linspace(peak_lo, peak_hi, 1000)
y_fit = gauss_bg(t_fit, *popt)

plt.figure(figsize=(8,4))
plt.step(centres, counts, where="mid", label="Data", color = 'black')
plt.grid(alpha = 0.3, zorder = 0)
plt.plot(t_fit, y_fit, label=f"Gaussian fit", color = 'r', zorder = 2)
plt.xlim(peak_lo - 5, peak_hi + 5)
plt.title('Prompt peak fit', fontsize = 20)
plt.xlabel(r"$t_\gamma - t_\alpha$ [ns]", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.legend(fontsize = 15)
plt.tight_layout()
plt.savefig('PeakWidthCFD.png')
plt.show()


# In[12]:


# half life

halflife = tau * np.log(2)
halflife_err = dtau * np.log(2)

print(f"Half Life = {halflife:.2f} ± {halflife_err:.2f} ns")


# In[13]:


# defining range for 2D hist
a_min = np.min(a_cond_V)
a_max = np.max(a_cond_V)

g_min = np.min(g_cond_V)
g_max = np.max(g_cond_V)

print(a_min, g_min)


# In[14]:


# 2d histogram
# coincidences
plt.figure(figsize=(8,7))

plt.hist2d(a_cond_V, g_cond_V, bins = 200, range = [[a_min, 0], [g_min, 0]], cmin = 1, cmax = 100)

plt.colorbar(label = 'Counts')
#plt.clim(1, np,max(h[0]))
plt.xlabel('Alpha peak amplitude (V)', fontsize = 20)
plt.ylabel('Gamma peak amplitude (V)', fontsize = 20)
plt.title('Alpha vs Gamma amplitudes (Coincidences)', fontsize = 20)
plt.savefig('2dhist_CFD.png')
plt.show()


# In[ ]:




