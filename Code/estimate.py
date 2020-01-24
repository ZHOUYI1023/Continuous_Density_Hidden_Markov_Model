import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def norm(mu, sigma, x):
    return 1.0 / np.sqrt(2 * np.pi * sigma) * np.exp(-1.0 / 2 * (x - mu) ** 2 / sigma)


class ContinuousHMM:
    def __init__(self, entry_prob, exit_prob, a, mu, sigma, observation, max_iter=20, threshold =1e-08):
        self.entry_prob = np.asarray(entry_prob)
        self.exit_prob = np.asarray(exit_prob)
        self.a = np.asarray(a)
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        self.observation = np.asarray(observation)
        self.N = a.shape[0]
        self.T = observation.shape[0]
        self.forward = np.zeros([self.N, self.T])
        self.backward = np.zeros([self.N, self.T])
        self.max_iter = max_iter
        self.threshold = threshold

    def getForwardProb(self):
        forward = np.zeros([self.N, self.T])
        for i in range(self.N):
            forward[i, 0] = self.entry_prob[i] * norm(self.mu[i], self.sigma[i], self.observation[0])
        for t in range(1, self.T):
            for i in range(self.N):
                for j in range(self.N):
                    forward[i, t] += forward[j, t - 1] * self.a[j, i] \
                                     * norm(self.mu[i], self.sigma[i], self.observation[t])
        forward_prob = sum(forward[:, -1] * self.exit_prob)
        self.forward = forward
        return forward_prob

    def getBackwardProb(self):
        backward = np.zeros([self.N, self.T])
        for i in range(self.N):
            backward[i, -1] = self.exit_prob[i]
        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    backward[i, t] += backward[j, t + 1] * self.a[i, j] \
                                      * norm(self.mu[j], self.sigma[j], self.observation[t + 1])

        backward_prob = 0
        for i in range(self.N):
            backward_prob += self.entry_prob[i] * backward[i, 0] \
                             * norm(self.mu[i], self.sigma[i], self.observation[0])
        self.backward = backward
        return backward_prob

    def getDecodingPath(self):
        viterbi = np.zeros([self.N, self.T])
        back_pointer = np.zeros([self.N, self.T], dtype=int)
        best_path = np.zeros(self.T, dtype=int)
        for i in range(self.N):
            viterbi[i, 0] = self.entry_prob[i] * norm(self.mu[i], self.sigma[i], self.observation[0])
        for t in range(1, self.T):
            for i in range(self.N):
                state_list = []
                for j in range(self.N):
                    state_list.append(viterbi[j, t - 1] * self.a[j, i])
                back_pointer[i, t] = np.argmax(state_list)
                viterbi[i, t] = viterbi[back_pointer[i, t], t - 1] * self.a[back_pointer[i, t], i] \
                                * norm(self.mu[i], self.sigma[i], self.observation[t])
        best_path_pointer = np.argmax(viterbi[:, -1] * self.exit_prob)
        best_path_prob = viterbi[best_path_pointer, -1] * self.exit_prob[best_path_pointer]
        best_path[-1] = best_path_pointer
        for t in range(self.T - 1, 0, -1):
            best_path[t - 1] = back_pointer[best_path[t], t]
        return best_path, best_path_prob, back_pointer

    def doTraining(self):
        data1 = [self.a[0,0]]
        data2 = [self.a[0,1]]
        data3 = [self.a[1,0]]
        data4 = [self.a[1,1]]
        data5 = [self.entry_prob[0]]
        data6 = [self.entry_prob[1]]
        data7 = [self.exit_prob[0]]
        data8 = [self.exit_prob[1]]
        data9 = [self.mu[0]]
        data10 = [self.mu[1]]
        data11 = [self.sigma[0]]
        data12 = [self.sigma[1]]
        for ind in range(self.max_iter):
            norm1 = self.getForwardProb()
            norm2 = self.getBackwardProb()
            ksi = np.zeros([self.N, self.N, self.T - 1])
            gamma = np.zeros([self.N, self.T])
            for t in range(0, self.T - 1):
                for i in range(self.N):
                    for j in range(self.N):
                        ksi[i, j, t] = self.forward[i, t] * self.a[i, j] \
                                       * norm(self.mu[j], self.sigma[j], self.observation[t + 1]) \
                                       * self.backward[j, t + 1] / norm1
            for t in range(self.T):
                for i in range(self.N):
                    gamma[i, t] = self.forward[i, t] * self.backward[i, t] / norm1
            for i in range(self.N):
                for j in range(self.N):
                    #self.a[i, j] = sum(ksi[i, j, :]) / sum(sum(ksi[i, :, :]))
                    self.a[i, j] = sum(ksi[i, j, :]) / sum(gamma[i, :])
                self.exit_prob[i] = 1 - sum(self.a[i, :])
            mu_old = np.copy(self.mu)
            for i in range(self.N):
                self.mu[i] = sum(gamma[i, :] * self.observation) / sum(gamma[i, :])
                self.sigma[i] = sum(gamma[i, :] * (self.observation - mu_old[i]) ** 2) / sum(gamma[i, :])
            self.entry_prob = gamma[:,0]
            print(self.mu)
            print(self.sigma)
            data1.append(self.a[0,0])
            data2.append(self.a[0,1])
            data3.append(self.a[1,0])
            data4.append(self.a[1,1])
            data5.append(self.entry_prob[0])
            data6.append(self.entry_prob[1])
            data7.append(self.exit_prob[0])
            data8.append(self.exit_prob[1])
            data9.append(self.mu[0])
            data10.append(self.mu[1])
            data11.append(self.sigma[0])
            data12.append(self.sigma[1])
            '''
            ax = sns.heatmap(gamma, annot=True, cmap="YlGnBu")
            ax.set_xlabel('time', fontsize=18)
            ax.set_ylabel('state', fontsize=18)
            ax.set_aspect(aspect=1.5)
            plt.show()
            '''
            if sum(mu_old - self.mu) < self.threshold:
                print(ind)
                break
        return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12
