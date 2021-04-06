import matplotlib.pyplot as plt

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

filename = 'pca_ret.txt'
X,Y1,Y2,Y3 = [],[],[],[]

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax1 = ax.twinx()

with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split()]
        X.append(value[0])
        Y1.append(value[1])
        Y2.append(value[2])
        Y3.append(value[3])
 
ln1 = ax.plot(X, Y1, 'y-.', label='Test loss (left)')
ln2 = ax1.plot(X, Y2, 'm--', label='Variance ratio (right)') 
ln3 = ax1.plot(X, Y3, 'r', label='MAL (right)') # mean absolute of latent layer

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='best', prop=font_legend)

ax.tick_params(labelsize=15)
ax1.tick_params(labelsize=15)
ax.set_xlabel('PCA components', font_axis)
# ax.set_ylabel(r'sum(Loss)', font_axis)

ax.set_ylim(0, ax.get_ylim()[1])
ax1.set_ylim(0, ax1.get_ylim()[1])

plt.subplots_adjust(bottom=0.12, right=0.88, left=0.12, top=0.99)

plt.show()
