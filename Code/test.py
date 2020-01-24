import numpy as np
from estimate import ContinuousHMM, norm
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

p = np.array([0.93, 0.07])
a = np.array([[0.74, 0.21], [0.08, 0.9]])
o = np.array([1.8, 2.6, 2.7, 3.3, 4.4, 5.4, 5.2])
e = np.array([0.05, 0.02])
mu = np.array([3.0, 5.0])
sigma = np.array([1.21, 0.25])
c = ContinuousHMM(p, e, a, mu, sigma, o)

if __name__ == "__main__":
    data1, data2, data3, data4, data5, data6, data7, data8, data9 ,data10, data11, data12 = c.doTraining()
    best_path, best_path_prob, back_pointer = c.getDecodingPath()
    print(back_pointer)
    fig, ax = plt.subplots()
    x = [0, 1, 2, 3, 4, 5, 6]
    ax.plot(x, [-1]+list(back_pointer[0, 1:]), 'bo-')
    ax.plot(x+[7,8], [-1]+list(back_pointer[1, 1:])+[1,2], 'ro-')
    ax.set_xlabel('time', fontsize=18)
    ax.set_ylabel('state', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_aspect(aspect=2.5)
    ax.grid(True)
    plt.axis('equal')
    plt.show()
    '''
    sigma = np.array([0.28923486, 0.18789497])
    mu = np.array([2.60233338, 5.00018899])
    fig, ax = plt.subplots()
    x = np.linspace(-0, 8, 120)
    ax.plot(x, norm(mu[0], sigma[0], x), label='b1')
    ax.plot(x, norm(mu[1], sigma[1], x), label='b2')
    for x in o:
        ax.plot(x, norm(mu[0], sigma[0], x), 'ro')
        ax.plot(x, norm(mu[1], sigma[1], x), 'bo')
    ax.legend(frameon=False, fontsize=18)
    ax.set_xlabel('value', fontsize=18)
    ax.set_ylabel('probability', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('./pdf.jpg', bbox_inches='tight')
    plt.show()
    '''
    '''
    prob = c.getBackwardProb()
    ax = sns.heatmap(c.backward, annot=True, cmap="YlGnBu", norm=LogNorm(vmin=c.forward.min(), vmax=c.forward.max()))
    ax.set_xlabel('time', fontsize=18)
    ax.set_ylabel('state', fontsize=18)
    ax.set_aspect(aspect=1.5)
    plt.show()
    '''
    #f = open("data.txt", "w")
    #np.savetxt(f, c.backward)

    '''
    # parameter
    fig, ax = plt.subplots()
    x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    #ax.plot(x, data1, 'x-',label='a00')
    #ax.plot(x, data2, 'x-',label='a01')
    #ax.plot(x, data3, 'x-',label='a10')
    #ax.plot(x, data4, 'x-',label='a11')
    #ax.plot(x, data5, 'x-', label='pi0')
    #ax.plot(x, data6, 'x-', label='pi1')
    ax.plot(x, data7, 'x-', label='eta0')
    ax.plot(x, data8, 'x-', label='eta1')
    #ax.plot(x, data9, 'x-', label='mu1')
    #ax.plot(x, data10, 'x-', label='mu2')
    #ax.plot(x, data11, 'x-', label='sigma1')
    #ax.plot(x, data12, 'x-', label='sigma2')
    ax.legend(frameon=False, fontsize=18)#, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('iteration', fontsize=18)
    ax.set_ylabel('value', fontsize=18)
    #ax.set_aspect(aspect=8.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    '''
    '''
    ## distribution
    fig, ax = plt.subplots()
    x = np.linspace(-0, 8, 120)
    for i in range(13):
        ax.plot(x, norm(data9[i], data11[i], x))
        ax.plot(x, norm(data10[i], data12[i], x))
        for y in o:
            ax.plot(y, norm(data9[i], data11[i], y), 'ro')
            ax.plot(y, norm(data10[i], data12[i], y), 'bo')
    ax.set_xlabel('value', fontsize=18)
    ax.set_ylabel('probability', fontsize=18)
    ax.legend(frameon=False, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.savefig('./pdf_all.jpg', bbox_inches='tight')
    plt.show()
    '''
    '''
    ## path
    best_path, best_path_prob, backpointer = c.getDecodingPath()
    fig, ax = plt.subplots()
    x = [1,2,3,4,5,6,7]
    ax.plot(x, backpointer[0,:], 'bo-')
    ax.plot(x, backpointer[1,:], 'ro-')
    ax.set_xlabel('time', fontsize=18)
    ax.set_ylabel('state', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_aspect(aspect=5.5)
    ax.grid(True)
    plt.axis('equal')
    plt.show()
    '''



