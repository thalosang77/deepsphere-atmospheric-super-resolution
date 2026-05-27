# ============================================================
# 2D spectral power analysis
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def radial_power_spectrum_2d(field_2d):
    """
    Computes radially averaged 2D Fourier power spectrum.
    """
    field = field_2d - np.mean(field_2d)
    fft = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(fft) ** 2

    ny, nx = field.shape
    y, x = np.indices((ny, nx))
    cy, cx = ny // 2, nx // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = r.max()
    spectrum = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for i in range(ny):
        for j in range(nx):
            spectrum[r[i, j]] += power[i, j]
            counts[r[i, j]] += 1

    spectrum = spectrum / np.maximum(counts, 1)
    return np.arange(max_r + 1), spectrum


sample_idx = 0
pressure_idx = 20

gt = direct_targets_q[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()
pred = direct_preds_q[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()
bil = baseline_preds_q[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()

k_gt, p_gt = radial_power_spectrum_2d(gt)
k_pred, p_pred = radial_power_spectrum_2d(pred)
k_bil, p_bil = radial_power_spectrum_2d(bil)

plt.figure(figsize=(9, 6))
plt.loglog(k_gt[1:], p_gt[1:], label="Ground Truth")
plt.loglog(k_bil[1:], p_bil[1:], label="Bilinear")
plt.loglog(k_pred[1:], p_pred[1:], label="DeepSphere")
plt.xlabel("Radial wavenumber")
plt.ylabel("Power")
plt.title(f"Radial Power Spectrum | sample={sample_idx}, pressure_channel={pressure_idx}")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

# High-frequency recovery score
hf_start = int(0.5 * len(k_gt))
hf_truth = np.sum(p_gt[hf_start:])
hf_bil = np.sum(p_bil[hf_start:])
hf_model = np.sum(p_pred[hf_start:])

print("CELL 18 PASSED")
print("High-frequency power:")
print("GT        :", hf_truth)
print("Bilinear  :", hf_bil)
print("DeepSphere:", hf_model)
print("Bilinear / GT  :", hf_bil / hf_truth)
print("DeepSphere / GT:", hf_model / hf_truth)
