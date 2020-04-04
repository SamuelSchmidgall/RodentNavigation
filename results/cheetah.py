import pickle
import random
import matplotlib.pyplot as plt

ax = plt.subplot(221)
for i in range(3):
	with open(f"./gated/cheetah/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])
	ax = plt.subplot(221)
	plt.plot(rewards, label=f"run {i}")
ax.title.set_text("Gated (cheetah)")
ax.legend()

ax = plt.subplot(222)
for i in range(3):
	with open(f"./nongated/cheetah/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])
	plt.plot(rewards, label=f"run {i}")
ax.title.set_text("Nongated (cheetah)")
ax.legend()

ax = plt.subplot(223)
for i in range(3):
	with open(f"./dropout/cheetah/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)

	rewards = []
	for j in range(len(m)):
		rewards.append(m[j][0])
	plt.plot(rewards, label=f"run {i}")
ax.title.set_text("Dropout (cheetah)")
ax.legend()

plt.show()