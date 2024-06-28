import matplotlib as mpl
from adjustText import adjust_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,estimate_bandwidth, \
      spectral_clustering,AgglomerativeClustering,DBSCAN,OPTICS, cluster_optics_dbscan, \
      Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA,KernelPCA,FastICA,FactorAnalysis 

# Parameters
colorbar_min=100
colorbar_max=220
label_good=140
label_bad=180
colorbar_step=20

### Data preparation
filename_train="data-summary.csv"
data_filename_train=pd.read_csv(filename_train)
name_train=data_filename_train.iloc[:,0]
y_train=data_filename_train.iloc[:,1]
x_train=data_filename_train.iloc[:,2:]

filename_new="new.csv"
data_filename_new=pd.read_csv(filename_new)
name_new=data_filename_new.iloc[:,0]
y_new=data_filename_new.iloc[:,1]
x_new=data_filename_new.iloc[:,2:]

# Normalization All data together
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x_all = pd.concat([x_train, x_new], axis=0)
x_all_normalized = min_max_scaler.fit_transform(x_all)
#x_train_normalized = x_all_normalized[:len(x_train)]
#x_new_normalized = x_all_normalized[len(x_train):]

# PCA dimensional reduction
method=PCA
pca=method(n_components=2,random_state=0)
pca_data=pca.fit_transform(x_all_normalized)
reduced_data=pca_data[:len(x_train),:] # Train data
reduced_data_new=pca_data[len(x_train):,:] # New data
pca_x_new=reduced_data_new[:,0]
pca_y_new=reduced_data_new[:,1]

# KMeans
for r in [13]: #np.arange(0,100):
    #print(r)
    method = Birch(threshold=0.2+0.01*r, n_clusters=2) # r=34, 37, 57, 62
    method.fit(reduced_data)
    
    plt.figure(figsize=(30, 15))
    h=0.02
    x_min, x_max = reduced_data[:, 0].min() - 0.5, reduced_data[:, 0].max() + 0.5
    y_min, y_max = reduced_data[:, 1].min() - 0.5, reduced_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = method.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.clf()
##    for m in np.arange(0,Z.shape[0]):
##        for n in np.arange(0,Z.shape[1]):
##            if Z[m][n] == 1:
##                plt.imshow(Z[m][n],c='r',alpha=0.3)
##            elif Z[m][n] == 0:
##                plt.imshow(Z[m][n],c='g',alpha=0.3)

    cmap=mpl.colors.ListedColormap(['royalblue','cyan'])
    plt.imshow(
        Z+1,interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=cmap, #cmap='viridis',vmin=0,vmax=3, #plt.cm.Paired, RdYlGn_r
        aspect="auto",
        origin="lower",
        alpha=0.3
    )
    #print(Z)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.array(y_train),cmap='RdYlGn_r',vmin=colorbar_min,vmax=colorbar_max,marker='o',s=300) #RdYlGn_r
    #print(np.array(y))
    ##plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=method.labels_,marker='o',s=250)
    ##
    texts = []
    bias_x=0.000;bias_y=0.000
    for xx, yy, s, n in zip(reduced_data[:, 0], reduced_data[:, 1], y_train,name_train):
        if s <= label_good:
            texts.append(plt.text(xx+bias_x, yy+bias_y, n,c='green',fontsize=20,horizontalalignment='left',verticalalignment='center'))
        elif s >= label_bad:
            texts.append(plt.text(xx+bias_x, yy+bias_y, n,c='red',fontsize=20,horizontalalignment='left',verticalalignment='center')) #,horizontalalignment='left',verticalalignment='center'
    #adjust_text(texts, only_move={'points':'x', 'texts':'x'},save_steps=False, arrowprops=dict(arrowstyle="->", color='b', lw=1))
            
    # Plot
    plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
    font1={
        'family':'Arial',
        'weight':'bold',
        'style':'normal',
        'size':25
        }
    plt.xlabel("PCA Feature 1",font1)
    plt.ylabel("PCA Feature 2",font1)
    bias_x2=0.300;bias_y2=0.300
    plt.xticks([]);plt.yticks([])
    #plt.xlim((min(reduced_data[:,0])-bias_x2,max(reduced_data[:,0])+bias_x2));plt.ylim((min(reduced_data[:,1])-bias_y2,max(reduced_data[:,1])+bias_y2))
    
    cb = plt.colorbar(pad=0.01)
    tick_locs = np.arange(colorbar_min, colorbar_max + colorbar_step, colorbar_step)
    cb.set_ticks(tick_locs)
    cb.ax.set_yticklabels(tick_locs, fontdict=font1)
    cb.set_label(r'DFT-computed $\Delta G_{min}^{\mathrm{‡}}$ (kJ/mol)', fontdict=font1)
    cb.ax.tick_params(labelsize=20)
    
    # New
    plt.scatter(pca_x_new,pca_y_new,c=np.array(y_new),cmap='RdYlGn_r',vmin=colorbar_min,vmax=colorbar_max,marker='*',s=1000)
    
    #texts2 = []
    for xx, yy, s, n in zip(pca_x_new, pca_y_new, y_new, name_new):
        if s <= label_good:
            #texts.append(plt.text(xx+bias_x, yy+bias_y, n,c='green',fontsize=20,horizontalalignment='left',verticalalignment='center',bbox=dict(facecolor='green', alpha=0.1)))
            texts.append(plt.text(xx + bias_x, yy + bias_y, n, c='green', fontsize=20,
                      horizontalalignment='left', verticalalignment='center',
                      bbox=dict(facecolor='green', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.5', linewidth=2)))


        elif s >= label_bad:
            texts.append(plt.text(xx + bias_x, yy + bias_y, n, c='red', fontsize=20,
                      horizontalalignment='left', verticalalignment='center',
                      bbox=dict(facecolor='red', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.5', linewidth=2)))

    adjust_text(texts, only_move={'points':'x', 'texts':'x'},save_steps=False, arrowprops=dict(arrowstyle="->", color='b', lw=1))
    
    plt.savefig('Gmin_plot.png',bbox_inches='tight')
    #plt.show()


