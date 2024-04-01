import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# with open('csv/panpanpanW.csv', 'r') as source_file, open('csv/panpanpanW_new.csv', 'w', newline='') as destination_file:
#     reader = csv.reader(source_file)
#     writer = csv.writer(destination_file)

#     for i, row in enumerate(reader):
#         if i == 0 or i % 2 == 1:
#             writer.writerow(row)

with open('csv/panpanpan.csv', 'r') as source_file, open('csv/panpanpan_new.csv', 'w', newline='') as destination_file:
    reader = csv.reader(source_file)
    writer = csv.writer(destination_file)

    last_row_time = None
    current_row_time = None
    for i, row in enumerate(reader):
        if i == 0:
            # Write the header row as is
            writer.writerow(row)
            continue
        if i % 2 == 1:
            current_row_time = float(row[0])
            if last_row_time is not None:
                # Subtract the time value from the previous row from the current row
                row[0] = current_row_time - last_row_time

            # Write the row to the new CSV file
            writer.writerow(row)

            # Update the last row time value
            last_row_time = current_row_time



with open('csv/panpanpan_new.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    data = list(reader)

x_index = headers.index('x')
y_index = headers.index('y')
z_index = headers.index('z')
time_index = headers.index('time')

time = np.array([float(row[time_index]) for row in data])
x = np.array([float(row[x_index]) for row in data])
y = np.array([float(row[y_index]) for row in data])
z = np.array([float(row[z_index]) for row in data])

plt.figure(figsize=(10, 6))
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.plot(z, label='z')
plt.legend()
plt.show()


fx_index = headers.index('fx')
fy_index = headers.index('fy')
fz_index = headers.index('fz')

fx = np.array([float(row[fx_index]) for row in data])
fy = np.array([float(row[fy_index]) for row in data])
fz = np.array([float(row[fz_index]) for row in data])


plt.figure(figsize=(10, 6))
plt.plot(fx, label='fx')
plt.plot(fy, label='fy')
plt.plot(fz, label='fz')
plt.legend()
plt.show()