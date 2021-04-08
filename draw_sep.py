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
data_id = np.load('pic_data/data_id.npy', allow_pickle=True).item()

for i in range(2228, 2461): ## modify to the real range
    fig = plt.figure(1,figsize=(8,10))
    ax = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)

    data = np.loadtxt('pic_data/data'+str(i)+'.txt')
    
    print(data_id[i+1])
    y1 = list(data[0]/np.amax(data[0]))
    y2 = list(data[1]/np.amax(data[1]))
    y3 = list(data[2]/np.amax(data[2]))
    
    x = np.array(list(range(len(y1)))) * 0.4 + 800
    
    ax.plot(x, y2, label='E2E_PCA_PCAI_IR_' + data_id[i+1], alpha=1.0)
    ax1.plot(x, y1, label='DFT_IR_' + data_id[i+1], alpha=1.0)
    ax2.plot(x, y3, label='DFT_PCA_PCAI_IR_' + data_id[i+1], alpha=1.0)
    
    ax.tick_params(labelsize=15)
    ax.legend(loc='best', prop=font_legend)
    ax1.tick_params(labelsize=15)
    ax1.legend(loc='best', prop=font_legend)
    ax2.tick_params(labelsize=15)
    ax2.legend(loc='best', prop=font_legend)

    # ax.grid(True)
    
    ax.set_ylabel(r'Intensity, A.U.', font_axis)
    ax1.set_ylabel(r'Intensity, A.U.', font_axis)
    ax2.set_ylabel(r'Intensity, A.U.', font_axis)
    ax.set_xlabel(r'$Wave\ Number, cm^{-1}$', font_axis)
    ax1.set_xlabel(r'$Wave\ Number, cm^{-1}$', font_axis)
    ax2.set_xlabel(r'$Wave\ Number, cm^{-1}$', font_axis)

    ax2.set_ylim(-1.1, 1.1)

    plt.subplots_adjust(bottom=0.12, right=0.99, left=0.11, top=0.99)
    plt.show()
