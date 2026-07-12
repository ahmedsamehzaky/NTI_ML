import numpy as np

fitness_data = np.array([
    [5000, 200, 6],
    [7000, 250, 7],
    [6500, 230, 6],
    [8000, 300, 8],
    [9000, 320, 7],
    [7500, 270, 6],
    [12000, 400, 5]
])

print(f"Number of dimensions : {fitness_data.ndim}")
print(f"Shape of the array: {fitness_data.shape}")
print(f"Total number of elements: {fitness_data.size}")

steps = fitness_data[:, 0]
calories = fitness_data[:, 1]
sleep = fitness_data[:, 2]
first_3_days = fitness_data[:3, :]
steps_and_calories = fitness_data[:, 0:2]

steps_reshaped = steps.reshape(7, 1)
full_reshaped = fitness_data.reshape(3, 7)
print(f"Steps reshaped to: {steps_reshaped.shape}")
print(f"Full data reshaped to: {full_reshaped.shape}")

days_high_steps = fitness_data[fitness_data[:, 0] > 8000]
days_low_sleep = fitness_data[fitness_data[:, 2] < 6]
days_high_calories = fitness_data[fitness_data[:, 1] > 300]
print(f"Days with steps > 8000:\n {days_high_steps}")

avg_steps = np.mean(steps)
max_steps = np.max(steps)
min_sleep = np.min(sleep)
std_calories = np.std(calories)
range_steps =np.max(steps) - np.min(steps)

print(f"Average steps: {avg_steps}")
print(f"Maximum steps: {max_steps}")
print(f"Minimum sleep hours: {min_sleep}")
print(f"Standard deviation of calories: {std_calories}")
print(f"Range of steps: {range_steps}")