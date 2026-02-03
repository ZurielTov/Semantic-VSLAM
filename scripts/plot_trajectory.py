#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Load trajectory data
data = np.loadtxt('../results/trajectory.txt')

frame = data[:, 0]
x_est = data[:, 1]
z_est = data[:, 2]
x_gt = data[:, 3]
z_gt = data[:, 4]

# Calculate error
error = np.sqrt((x_est - x_gt)**2 + (z_est - z_gt)**2)
mean_error = np.mean(error)
final_error = error[-1]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory plot (bird's eye view)
ax1.plot(x_gt, z_gt, 'g-', linewidth=2, label='Ground Truth')
ax1.plot(x_est,z_est, 'r--', linewidth=2, label='Estimated')
ax1.plot(x_gt[0],z_gt[0], 'go', markersize=10, label='Start')
ax1.plot(x_gt[-1],z_gt[-1], 'rx', markersize=10, label='End')
ax1.set_xlabel('X (meters)', fontsize=12)
ax1.set_ylabel('Z (meters)', fontsize=12)
ax1.set_title('Trajectory (Top View)', fontsize=14)
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Error plot
ax2.plot(frame, error, 'b-', linewidth=2)
ax2.axhline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}m')
ax2.set_xlabel('Frame', fontsize=12)
ax2.set_ylabel('Error (meters)', fontsize=12)
ax2.set_title(f'Trajectory Error\nFinal: {final_error:.2f}m', fontsize=14)
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig('../results/trajectory_plot.png', dpi=150)
print(f"✅ Plot saved to: ../results/trajectory_plot.png")
print(f"Mean error: {mean_error:.2f} meters")
print(f"Final error: {final_error:.2f} meters")
plt.show()