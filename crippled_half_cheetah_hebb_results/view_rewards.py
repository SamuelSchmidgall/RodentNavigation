import pickle
import random
import matplotlib.pyplot as plt

for i in range(1,4):
	with open(f"./run_{i}/save_rewardhebb{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])

	plt.plot(rewards, label=f"run {i}")

plt.legend()
plt.show()