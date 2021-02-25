import numpy as np
import matplotlib.pyplot as plt

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
data_id = np.load('data_id.npy', allow_pickle=True).item()

for i in range(100):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    data = np.loadtxt('data'+str(i)+'.txt')
    
    print(data_id[i+1])
    y1 = list(data[0]/np.amax(data[0]))
    y2 = list(data[1]/np.amax(data[1]))
    
    x = np.array(list(range(len(y1)))) * 0.4 + 800
    
    ax.plot(x, y2, label='E2E_IR_' + data_id[i+1], alpha=0.7)
    ax.plot(x, y1, label='DFT_IR_' + data_id[i+1], alpha=1.0)
    
    ax.tick_params(labelsize=15)
    ax.legend(loc='upper right', prop=font_legend)
    # ax.grid(True)
    
    ax.set_ylabel(r'Intensity, A.U.', font_axis)
    ax.set_xlabel(r'$Wave\ Number, cm^{-1}$', font_axis)



    plt.subplots_adjust(bottom=0.12, right=0.99, left=0.11, top=0.99)
    plt.show()
