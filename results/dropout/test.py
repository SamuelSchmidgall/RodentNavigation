import pickle

print("hopper")
for i in range(3):
	with open(f"./hopper/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)
	print(i, len(m))

print("cheetah")
for i in range(3):
	with open(f"./cheetah/rewards_{i}.pkl",'rb') as f:
		m = pickle.load(f)
	print(i, len(m))