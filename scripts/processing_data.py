import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Strong
# First case: N = 566 MB
data = pd.read_csv("./data/strong_first_case.csv", sep=";")
proc = np.array(data["# P"])
timeBySlot = np.array(data["Time by slot"])
timeByNode = np.array(data["Time by node"])
speedupBySlot = np.array(data["Speedup by slot"])
speedupByNode = np.array(data["Speedup by node"])

# Time
plt.figure(1)
plt.plot(proc, timeBySlot, '-b', label="By slot")
plt.plot(proc, timeByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 20])
plt.xlabel("# of Processors")
plt.ylabel('Time(s)')
plt.title("N = 566 MB")
plt.xticks(proc)
plt.yticks(np.arange(0, 21, 1.5))
plt.legend()
plt.grid()

# Speedup
plt.figure(2)
plt.plot(proc, speedupBySlot, '-b', label="By slot")
plt.plot(proc, speedupByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 6])
plt.xlabel("# of Processors")
plt.ylabel('Speedup')
plt.title("N = 566 MB")
plt.yticks(np.arange(0, 7, 0.5))
plt.xticks(proc)
plt.legend()
plt.grid()

# Second case: N = 1.2 GB

data = pd.read_csv("./data/strong_second_case.csv", sep=";")
proc = np.array(data["# P"])
timeBySlot = np.array(data["Time by slot"])
timeByNode = np.array(data["Time by node"])
speedupBySlot = np.array(data["Speedup by slot"])
speedupByNode = np.array(data["Speedup by node"])

# Time
plt.figure(3)
plt.plot(proc, timeBySlot, '-b', label="By slot")
plt.plot(proc, timeByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 30])
plt.xlabel("# of Processors")
plt.ylabel('Time(s)')
plt.title("N = 1.2 GB")
plt.xticks(proc)
plt.yticks(np.arange(0, 31, 1.5))
plt.legend()
plt.grid()

# Speedup
plt.figure(4)
plt.plot(proc, speedupBySlot, '-b', label="By slot")
plt.plot(proc, speedupByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 8])
plt.xlabel("# of Processors")
plt.ylabel('Speedup')
plt.title("N = 1.2 GB")
plt.yticks(np.arange(0, 9, 0.5))
plt.xticks(proc)
plt.legend()
plt.grid()

# Weak
# First case: N = 17.7 MB
data = pd.read_csv("./data/weak_first_case.csv", sep=";")
proc = np.array(data["# P"])
timeBySlot = np.array(data["Time by slot"])
timeByNode = np.array(data["Time by node"])
efficiencyBySlot = np.array(data["Efficiency by slot"])
efficiencyByNode = np.array(data["Efficiency by node"])

# Time
plt.figure(5)
plt.plot(proc, timeBySlot, '-b', label="By slot")
plt.plot(proc, timeByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 6])
plt.xlabel("# of Processors")
plt.ylabel('Time(s)')
plt.title("Workload of 17.7 MB per process")
plt.xticks(proc)
plt.yticks(np.arange(0, 7, 0.5))
plt.legend()
plt.grid()

# Efficiency
plt.figure(6)
plt.plot(proc, efficiencyBySlot, '-b', label="By slot")
plt.plot(proc, efficiencyByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 1])
plt.xlabel("# of Processors")
plt.ylabel('Efficiency')
plt.title("Workload of 17.7 MB per process")
plt.xticks(proc)
plt.yticks(np.arange(0, 1, 0.15))
plt.legend()
plt.grid()

# Second case: N = 54 MB
data = pd.read_csv("./data/weak_second_case.csv", sep=";")
proc = np.array(data["# P"])
timeBySlot = np.array(data["Time by slot"])
timeByNode = np.array(data["Time by node"])
efficiencyBySlot = np.array(data["Efficiency by slot"])
efficiencyByNode = np.array(data["Efficiency by node"])

# Time
plt.figure(7)
plt.plot(proc, timeBySlot, '-b', label="By slot")
plt.plot(proc, timeByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 6])
plt.xlabel("# of Processors")
plt.ylabel('Time(s)')
plt.title("Workload of 54 MB per process")
plt.xticks(proc)
plt.yticks(np.arange(0, 7, 0.5))
plt.legend()
plt.grid()

# Efficiency
plt.figure(8)
plt.plot(proc, efficiencyBySlot, '-b', label="By slot")
plt.plot(proc, efficiencyByNode, '-r', label="By node")
plt.axis([proc[0], proc[len(proc) - 1], 0, 1])
plt.xlabel("# of Processors")
plt.ylabel('Efficiency')
plt.title("Workload of 54 MB per process")
plt.xticks(proc)
plt.yticks(np.arange(0, 1, 0.10))
plt.grid()
plt.legend()


plt.show()