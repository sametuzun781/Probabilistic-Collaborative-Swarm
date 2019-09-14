import seaborn as sns
import matplotlib.pyplot as plt

def plt_fcn(N_time, N_of_figures, counter, img, Fig_size_scale, res_dist, total_variation, Delta_error, N1, ita0_N1_ind):

    save_count = int(N_time/N_of_figures)
    for i in range(int(counter/save_count)):
        plt.figure(num=i, figsize=(img.shape[1]/Fig_size_scale,img.shape[0]/Fig_size_scale))
        sns_plot = sns.heatmap(res_dist[:,:,i*save_count], vmin=0, vmax=int(N1 / len(ita0_N1_ind)))
        sns_plot = sns_plot.get_figure()
        print(i)
        fig_name = str(i) + '.png'
        sns_plot.savefig(fig_name)

    plt.figure(figsize=(10,5))
    plt.plot(total_variation)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('Total Variation', fontsize=12)
    plt.savefig('Total Variation')
    # plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(Delta_error)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('Standard Deviation of the Swarm', fontsize=12)
    plt.savefig('Standard Deviation of the Swarm')
    # plt.show()