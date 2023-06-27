import numpy as np
b = [0.3,0.2,0.1,0.06,0.03]
a=np.array(b)
d = np.argsort(a)[::-1]
index = []
for v in np.argsort(a)[::-1][:3]:
    print(b[v])
    index.append(v)

