import pickle
import random
import matplotlib.pyplot as plt

ax = plt.subplot(221)
for i in range(3):
	with open(f"./gated/hopper/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])
	ax = plt.subplot(221)
	plt.plot(rewards, label=f"run {i}")
ax.title.set_text("Gated (hopper)")
ax.legend()

ax = plt.subplot(222)
for i in range(3):
	with open(f"./nongated/hopper/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])
	plt.plot(rewards, label=f"run {i}")
ax.title.set_text("Nongated (hopper)")
ax.legend()

ax = plt.subplot(223)
for i in range(3):
	with open(f"./dropout/hopper/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])
	plt.plot(rewards, label=f"run {i}")
ax.title.set_text("Dropout (hopper)")
ax.legend()

plt.show()