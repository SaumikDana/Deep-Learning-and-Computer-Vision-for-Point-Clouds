import matplotlib.pyplot as plt
import numpy as np

# Process steps and their times
process_steps = [
    'Encoding', 'Triangulation', 'Filtering', 'Normal Computation', 
    'Feature Extraction', 'Random Sampling', 'Fine Registration'
]
times = [0.5, 0.11, 0.01, 2, 0.05, 0.95, 1]

# Sorting the steps and times based on times in descending order
sorted_steps = [step for _, step in sorted(zip(times, process_steps), reverse=True)]
sorted_times = sorted(times, reverse=True)

# Calculating the cumulative sum of times
cumulative_sum = np.cumsum(sorted_times)

# Creating the Pareto chart with tilted labels
plt.rcParams.update({'font.size': 8})  # Reducing font size
fig, ax1 = plt.subplots()

ax1.bar(sorted_steps, sorted_times, color='blue')
# ax1.set_xlabel('Process Steps')
ax1.set_ylabel('Time (seconds)', color='blue')
ax1.tick_params('y', colors='blue')
plt.xticks(rotation=45)  # Tilting the x-axis labels

# Adding a line plot for the cumulative sum
ax2 = ax1.twinx()
ax2.plot(sorted_steps, cumulative_sum, color='red', marker='o')
ax2.set_ylabel('Cumulative Time (seconds)', color='red')
ax2.tick_params('y', colors='red')

# plt.title('Pareto Chart of Process Steps and Times')
plt.tight_layout()  # Adjust layout to accommodate tilted labels
plt.show()
