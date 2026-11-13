from sklearn.preprocessing import StandardScaler

data = [[2],[4],[6],[8],[10]]

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("Original Data:")
print(data)

print("Mean:", scaler.mean_)
print("Standard Deviation:", scaler.scale_)

print("Standard Scaled Data:")
print(scaled_data)