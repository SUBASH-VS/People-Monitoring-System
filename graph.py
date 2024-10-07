import matplotlib.pyplot as plt
import time
import datetime

# Initialize lists to store time (x-axis) and values (y-axis)
x = []
y = []

# Record data for 10 seconds
for i in range(10):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")  # Get current time in HH:MM:SS format
    x.append(current_time)
    y.append(i + 1)  # Increment y-value each second
    time.sleep(1)  # Wait for 1 second

# Create a line plot
plt.plot(x, y, marker='o', label='Increment per second')

# Add labels and title
plt.xlabel('Time (HH:MM:SS)')
plt.ylabel('Value (Y)')
plt.title('Value Increment Over Time')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display a legend
plt.legend()

# Show the graph
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
