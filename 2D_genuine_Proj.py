import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize_scalar

#Apply projections for calculating maximum norm based genuine adversarial accuracy

np.random.seed(0)

def noise_gen(sig,n,ord=2):
	#Noise generator whose distribution is Gaussian RBF (radial basis function)
    noise_ = np.random.normal(scale=sig, size=[n, 2])

    noise_norm = np.linalg.norm(noise_, axis=1, keepdims=True)
    noise_pos = np.random.uniform(low=-1.0, high=1.0, size=[n, 1])

    if ord == 1:
        my_noise = noise_norm * np.concatenate(
            [1 - 2 * np.abs(noise_pos), np.abs(2 * noise_pos + 1) - np.abs(2 * noise_pos - 1) - 2 * noise_pos], axis=1)  # p_ord=1
    elif ord == 2:
        my_noise = noise_norm * np.concatenate([np.cos(np.pi * noise_pos), np.sin(np.pi * noise_pos)], axis=1)  # p_ord=2
    elif ord == np.inf:
        my_noise = noise_norm * np.concatenate([-0.5 * np.abs(4 * noise_pos + 1) - 0.5 * np.abs(4 * noise_pos - 1) + 0.5 * np.abs(4 * noise_pos + 3) + 0.5 * np.abs(4 * noise_pos - 3) - 1,
                                                0.5 * np.abs(4 * noise_pos + 1) - 0.5 * np.abs(4 * noise_pos - 1) + 0.5 * np.abs(4 * noise_pos + 3) - 0.5 * np.abs(4 * noise_pos - 3) - 4 * noise_pos],axis=1)
    else:
        pass
    return my_noise



p_ord=1 #Choose order for lp norm. 1 or 2 or np.inf
print('p:',p_ord)
stab_eps=0.00001 #Small value that will be used for numerical stability

#Ring data
'''k=12
x1=2*np.array([[np.cos(i/k*2*np.pi),np.sin(i/k*2*np.pi)] for i in range(k)])
x2=np.array([[0,0]])
xy_lim=[-4,4]'''

#Square data
'''x1=np.array([[3,1],[3,-1],[3,3],[3,-3],[1,3],[1,-3],[-1,3],[-1,-3],[-3,3],[-3,-3],[-3,1],[-3,-1]])
x2=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
xy_lim=[-4,4]'''

#Example in the paper (in the supplementary section)
x1=np.array([[-1,-1],[-2,1],[2,-2],[1,1],[2,-1]])
x2=np.array([[0,0],[2,1],[1,-1]])
xy_lim=[-3.5,3.5]

#Linear
#x1=np.array([[-1,-1]])
#x2=np.array([[1,1]])

#x1=np.array([[-1,0]])
#x2=np.array([[1,0]])

#Sunset
'''np.random.seed(3)
x1=10*np.random.normal(size=[6,1])
x1=np.concatenate([np.cos(x1),1+np.sin(x1)],axis=1)
x2=2*np.random.normal(size=[6,2])
x2[:,1]=0.0
xy_lim=[-4,4]'''

x=np.concatenate([x1, x2],axis=0)

lin_number=100
x_lin = np.linspace(xy_lim[0], xy_lim[1], lin_number)
y_lin = np.linspace(xy_lim[0], xy_lim[1], lin_number)
X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
U_lin = np.concatenate((X_lin.reshape([-1,1]),Y_lin.reshape([-1,1])),axis=1)

sigma=2.5
epsilon=2.0
cm = plt.get_cmap('jet')
print('Sigma',sigma)

swc_gradual = 0 #Switch for plotting gradual 1-NN or standard 1-NN
swc_alpha = 1 #When applying projections, this switch will make to use a bit larger alpha (stab_eps*epsilon)

[n1, n2]=[len(x1), len(x2)]
n_x=n1+n2
if swc_gradual==1:
    diff1 = U_lin.reshape([1, len(U_lin), -1]) - x1.reshape([n1, 1, -1])
    diff2 = U_lin.reshape([1, len(U_lin), -1]) - x2.reshape([n2, 1, -1])
    diff1_norm = np.linalg.norm(diff1, axis=-1, ord=p_ord)
    diff2_norm = np.linalg.norm(diff2, axis=-1, ord=p_ord)
    nn1_norm = np.min(diff1_norm, axis=0)  # Nearest neighbor distance to class 1
    nn2_norm = np.min(diff2_norm, axis=0)  # Nearest neighbor distance to class 2
    prob_result = (1 / nn1_norm) / ((1 / nn1_norm) + (1 / nn2_norm))  # Use 1/x for normalization in gradual 1-NN
else:
    neigh = KNeighborsClassifier(n_neighbors=1, p=p_ord)
    neigh.fit(np.concatenate([x1, x2], 0), [1]*n1+[0]*n2)
    prob_result = neigh.predict_proba(U_lin)
    prob_result = prob_result[:, 1]
    prob_result[np.where(prob_result < 0.5)[0]] = 0
    prob_result[np.where(prob_result > 0.5)[0]] = 1

    neigh = KNeighborsClassifier(n_neighbors=1, p=p_ord)
    neigh.fit(np.concatenate([x2, x1], 0), [0] * n2 + [1] * n1)  # Flipped
    prob_result2 = neigh.predict_proba(U_lin)
    prob_result2 = prob_result2[:, 1]
    prob_result2[np.where(prob_result2 < 0.5)[0]] = 0
    prob_result2[np.where(prob_result2 > 0.5)[0]] = 1

    prob_result = (prob_result + prob_result2) / 2

plt.figure(1)
plt.contourf(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), 20, cmap='jet')
plt.contour(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), levels=[0.5], colors='black', linewidths=2, linestyles='dashed')
plt.scatter(x1[:,0], x1[:,1],s=20,c='r')
plt.scatter(x2[:, 0], x2[:, 1],s=20, c='b')
plt.xlim(xy_lim)
plt.ylim(xy_lim)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

def ndarray_l1_norm(x):
    x_norm = np.sum(np.abs(x), axis=tuple(range(len(x.shape))[1:]), keepdims=True)
    return x_norm

def ndarray_norm(x):
    x_norm = np.sum(x ** 2, axis=tuple(range(len(x.shape))[1:]), keepdims=True) ** 0.5
    return x_norm

def gen_Proj(x_prime,x_nat,eps):
    #Apply projection for calculating genuine adversarial accuracy based on maximum norm
    #Shape of x_prime: [N, d], Shape of x_nat: [N, d]

    #First, apply projection based on lp ball (using maximum norm)
    if p_ord==1:
        alpha= np.maximum(0,1-eps/(ndarray_l1_norm(x_prime - x_nat) + stab_eps))
        x_prime = x_prime+alpha*(x_nat-x_prime)
    elif p_ord==2:
        alpha= np.maximum(0,1-eps/(ndarray_norm(x_prime - x_nat) + stab_eps))
        x_prime = x_prime+alpha*(x_nat-x_prime)
    elif p_ord==np.inf:
        x_prime = np.clip(x_prime, x_nat - eps, x_nat + eps)

    history = np.zeros([len(x_prime), len(x_prime), 2])
    history[:,0,:]=x_prime
    history2 = np.zeros([len(x_prime), len(x_prime), 2])
    d_argsort = np.zeros([len(x_prime),len(x_prime)-1],dtype='int')
    #Second, apply projection based on Voronoi cell
    for i, xi in enumerate(x_prime):
        tmp_inds = np.setdiff1d(np.arange(len(x_nat)), [i])
        tmp_d = np.linalg.norm(x_nat[i] - x_nat[tmp_inds], axis=1, ord=p_ord)
        tmp_d=np.insert(tmp_d,i,0.0)
        d_argsort_=np.argsort(tmp_d)
        d_argsort[i] = np.delete(d_argsort_, 0)
        # Apply projection so that new point is closest (or maybe tied) to x_i (i.e. x_nat[i])
        for jj, j in enumerate(d_argsort[i]):
            if p_ord == 2:
                alpha = np.maximum(0,  (x_nat[i]-x_nat[j]).dot((x_nat[i]+x_nat[j])/2-xi)/(np.linalg.norm(x_nat[i]-x_nat[j], ord=p_ord) + stab_eps)**2)
            else:
                #When p_ord is not 2, use optimizer
                def loss(a):
                    d_i = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_nat[j]) - x_nat[i], ord=p_ord)
                    d_j = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_nat[j]) - x_nat[j], ord=p_ord)
                    return np.maximum(d_i, d_j)
                alpha_max = np.linalg.norm(x_prime[i] - x_nat[i], ord=p_ord) / (np.linalg.norm(x_nat[i] - x_nat[j], ord=p_ord) + stab_eps)
                tmp_opt = minimize_scalar(loss, method='bounded', bounds=(0,alpha_max),tol=1e-5)
                alpha=tmp_opt.x
            if swc_alpha==1:
                alpha+=stab_eps*eps
            x_prime[i]=x_prime[i]+alpha*(x_nat[i]-x_nat[j])
            history[i,jj+1,:]=x_prime[i]
        history2[i, 0, :] = x_prime[i]

        # Find a new point that lies in the outside of Voronoi boundary
        for jj, j in enumerate(d_argsort[i]):
            d_i = np.linalg.norm(x_prime[i] - x_nat[i], ord=p_ord)
            d_j = np.linalg.norm(x_prime[i] - x_nat[j], ord=p_ord)
            if d_i+stab_eps < d_j:
                alpha2 = 0.0
            else:
                if p_ord == 2:
                    alpha2 = np.maximum(0,  (x_nat[i]-x_nat[j]).dot((x_nat[i]+x_nat[j])/2-x_prime[i])/((x_nat[i]-x_nat[j]).dot(x_nat[i]-x_prime[i])+ 0.00001**2))
                else:
                    # When p_ord is not 2, use optimizer
                    def loss2(a):
                        d_i = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_prime[i]) - x_nat[i], ord=p_ord)
                        d_j = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_prime[i]) - x_nat[j], ord=p_ord)
                        return np.abs(d_i-d_j)-stab_eps*a
                    tmp_opt2 = minimize_scalar(loss2, method='bounded', bounds=(0,1),tol=1e-5)
                    alpha2=tmp_opt2.x
            if swc_alpha==1:
                alpha2 += stab_eps * eps
            x_prime[i] = x_prime[i] + alpha2 * (x_nat[i] - x_prime[i])
            history2[i, jj + 1, :] = x_prime[i]
    return x_prime, history, history2, d_argsort

def gen_Proj2(x_prime, x_nats, eps):
    #Apply projection for calculating genuine adversarial accuracy based on maximum norm
    #The difference with 'gen_Proj' is that this function use only 'm' nearest neighbors for projection to roughly apply projection
    # (but faster and require less memory)
    #Shape of x_prime: [N, d], Shape of x_nats: [N, m+1, d] where 'm' is the number of nearest samples used (m+1: because it includes the sample itself)

    x_nat=x_nats[:,0,:]
    m=len(x_nats[0,:,0])-1
    d=x_prime.shape[1]

    #First, apply projection based on lp ball (using maximum norm)
    if p_ord==1:
        alpha= np.maximum(0,1-eps/(ndarray_l1_norm(x_prime - x_nat) + stab_eps))
        x_prime = x_prime+alpha*(x_nat-x_prime)
    elif p_ord==2:
        alpha= np.maximum(0,1-eps/(ndarray_norm(x_prime - x_nat) + stab_eps))
        x_prime = x_prime+alpha*(x_nat-x_prime)
    elif p_ord==np.inf:
        x_prime = np.clip(x_prime, x_nat - eps, x_nat + eps)

    history = np.zeros([len(x_prime), m+1, d])
    history[:,0,:]=x_prime
    history2 = np.zeros([len(x_prime), m+1, d])
    #Second, apply projection based on Voronoi cell
    for i, xi in enumerate(x_prime):
        # Apply projection so that new point is closest (or maybe tied) to x_i (i.e. x_nat[i])
        for j in range(m):
            if p_ord == 2:
                alpha = np.maximum(0, (x_nat[i]-x_nats[i,j+1,:]).dot((x_nat[i]+x_nats[i,j+1,:])/2-xi)/(np.linalg.norm(x_nat[i]-x_nats[i,j+1,:], ord=p_ord)+stab_eps)**2)
            else:
                #When p_ord is not 2, use optimizer
                def loss(a):
                    d_i = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_nats[i,j+1,:]) - x_nat[i], ord=p_ord)
                    d_j = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_nats[i,j+1,:]) - x_nats[i,j+1,:], ord=p_ord)
                    return np.maximum(d_i, d_j)
                alpha_max = np.linalg.norm(x_prime[i] - x_nat[i], ord=p_ord) / (np.linalg.norm(x_nat[i] - x_nats[i,j+1,:], ord=p_ord) + stab_eps)
                tmp_opt = minimize_scalar(loss, method='bounded', bounds=(0,alpha_max),tol=1e-5)
                alpha=tmp_opt.x
            if swc_alpha==1:
                alpha+=stab_eps*eps
            x_prime[i]=x_prime[i]+alpha*(x_nat[i]-x_nats[i,j+1,:])
            history[i,j+1,:]=x_prime[i]
        history2[i, 0, :] = x_prime[i]

        # Find a new point that lies in the outside of Voronoi boundary
        for j in range(m):
            d_i = np.linalg.norm(x_prime[i] - x_nat[i], ord=p_ord)
            d_j = np.linalg.norm(x_prime[i] - x_nats[i,j+1,:], ord=p_ord)
            if d_i+stab_eps < d_j:
                alpha2 = 0.0
            else:
                if p_ord == 2:
                    alpha2 = np.maximum(0, (x_nat[i]-x_nats[i,j+1,:]).dot((x_nat[i]+x_nats[i,j+1,:])/2-x_prime[i])/((x_nat[i]-x_nats[i,j+1,:]).dot(x_nat[i]-x_prime[i])+ 0.00001**2))
                else:
                    # When p_ord is not 2, use optimizer
                    def loss2(a):
                        d_i = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_prime[i]) - x_nat[i], ord=p_ord)
                        d_j = np.linalg.norm(x_prime[i] + a * (x_nat[i] - x_prime[i]) - x_nats[i,j+1,:], ord=p_ord)
                        return np.abs(d_i-d_j)-stab_eps*a
                    tmp_opt2 = minimize_scalar(loss2, method='bounded', bounds=(0,1), tol=1e-5)
                    alpha2=tmp_opt2.x
            if swc_alpha==1:
                alpha2 += stab_eps * eps
            x_prime[i] = x_prime[i] + alpha2 * (x_nat[i] - x_prime[i])
            history2[i, j + 1, :] = x_prime[i]
    return x_prime, history, history2



t1=time.time()

n_noise= 200 #Number of noises
x1_primes=np.ones([0,2])
x2_primes=np.ones([0,2])
x_primes=np.ones([0,2])
historys_2=np.ones([0,n_x,2])
history2s_2=np.ones([0,n_x,2])
d_argsorts=np.ones([0,n_x-1],dtype='int')
for j in range(n_noise):
    x_noise= x + noise_gen(sigma, n1+n2, ord=p_ord) #Use noise added points instead of gradient added points to show example

    x_prime, history, history2, d_argsort = gen_Proj(x_noise, x, eps=epsilon)
    x_primes = np.concatenate([x_primes, x_prime], axis=0)
    historys_2 = np.concatenate([historys_2, history], axis=0)
    history2s_2 = np.concatenate([history2s_2, history2], axis=0)
    d_argsorts = np.concatenate([d_argsorts, d_argsort], axis=0)

t2=time.time()
print('Spend time:',t2-t1)

x1_inds=np.where(np.arange(len(x_primes))%n_x<n1)[0]
x2_inds=np.where(np.arange(len(x_primes))%n_x>=n1)[0]
x1_dists=np.linalg.norm(x_primes.reshape(-1,1,2)-x1.reshape([1,-1,2]),axis=-1,ord=p_ord)
x2_dists=np.linalg.norm(x_primes.reshape(-1,1,2)-x2.reshape([1,-1,2]),axis=-1,ord=p_ord)
x1_nn_dists=np.min(x1_dists,axis=-1)
x2_nn_dists=np.min(x2_dists,axis=-1)
x_VB_inds = np.where(x1_nn_dists==x2_nn_dists)[0] #Index of points in Voronoi boundary
print('Number of points in Voronoi boundary:',len(x_VB_inds))
print('Percentage of points in Voronoi boundary:',len(x_VB_inds)/len(x_primes)*100)

plt.figure(2)
plt.scatter(x1[:,0], x1[:,1],s=20,c='r')
plt.scatter(x2[:, 0], x2[:, 1],s=20, c='b')
plt.scatter(x_primes[x1_inds, 0], x_primes[x1_inds, 1],s=0.5, c='r')
plt.scatter(x_primes[x2_inds, 0], x_primes[x2_inds, 1],s=0.5, c='b')
plt.scatter(x_primes[x_VB_inds, 0], x_primes[x_VB_inds, 1],s=0.5, c='purple')
plt.xlim(xy_lim)
plt.ylim(xy_lim)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
if p_ord == np.inf:
    plt.savefig('Proj_points_linf.png', bbox_inches='tight', dpi=200)
else:
    plt.savefig('Proj_points_l{}.png'.format(str(p_ord)), bbox_inches='tight', dpi=200)
plt.show()

import matplotlib.cm as cm
colors = cm.Greens(np.linspace(0, 1, n_x))
markers = ['.','v','^','<','>','*','X']

### Visualize projection process ###
plt.figure(3)
for i in range(len(x)):
    tmp_inds = np.where(np.arange(len(x_primes)) % n_x == i)[0]
    tmp_diff_inds = np.setdiff1d(np.arange(len(x)), [i])
    for j in range(len(x)):
        plt.clf()
        plt.scatter(x1[:, 0], x1[:, 1], s=10, c='r')
        plt.scatter(x2[:, 0], x2[:, 1], s=10, c='b')
        plt.scatter(x[i, 0], x[i, 1], s=1, c='gray')
        sc = plt.scatter(historys_2[tmp_inds, j, 0], historys_2[tmp_inds, j, 1], s=1, color='k', marker=markers[j % 7])
        if j>0:
            plt.scatter(x[d_argsorts[tmp_inds][:,j-1], 0], x[d_argsorts[tmp_inds][:,j-1], 1], s=1, c='gray')
            plt.quiver(historys_2[tmp_inds, j-1, 0], historys_2[tmp_inds, j-1, 1],
                       historys_2[tmp_inds, j, 0]-historys_2[tmp_inds, j-1, 0], historys_2[tmp_inds, j, 1]-historys_2[tmp_inds, j-1, 1], scale_units='xy',scale=1,width =0.001,color='grey')
        plt.xlim(xy_lim)
        plt.ylim(xy_lim)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.gca().set_aspect('equal', adjustable='box')
        if p_ord==np.inf:
            plt.savefig('Proj_points_linf_{}_{}.png'.format(str(i + 1),str(j + 1)), bbox_inches='tight', dpi=200)
        else:
            plt.savefig('Proj_points_l{}_{}_{}.png'.format(str(p_ord),str(i + 1), str(j + 1)), bbox_inches='tight', dpi=200)

        plt.clf()
        plt.scatter(x1[:, 0], x1[:, 1], s=10, c='r')
        plt.scatter(x2[:, 0], x2[:, 1], s=10, c='b')
        plt.scatter(x[i, 0], x[i, 1], s=1, c='gray')
        sc = plt.scatter(history2s_2[tmp_inds, j, 0], history2s_2[tmp_inds, j, 1], s=1, color='k',
                         marker=markers[j % 7])
        if j > 0:
            plt.scatter(x[d_argsorts[tmp_inds][:, j - 1], 0], x[d_argsorts[tmp_inds][:, j - 1], 1], s=1, c='gray')
            plt.quiver(history2s_2[tmp_inds, j - 1, 0], history2s_2[tmp_inds, j - 1, 1],
                       history2s_2[tmp_inds, j, 0] - history2s_2[tmp_inds, j - 1, 0],
                       history2s_2[tmp_inds, j, 1] - history2s_2[tmp_inds, j - 1, 1], scale_units='xy', scale=1,
                       width=0.001, color='grey')
        plt.xlim(xy_lim)
        plt.ylim(xy_lim)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.gca().set_aspect('equal', adjustable='box')
        if p_ord == np.inf:
            plt.savefig('next_Proj_points_linf_{}_{}.png'.format(str(i + 1), str(j + 1)), bbox_inches='tight', dpi=200)
        else:
            plt.savefig('next_Proj_points_l{}_{}_{}.png'.format(str(p_ord), str(i + 1), str(j + 1)), bbox_inches='tight', dpi=200)


### Visualize simple case of properly applied adversarial training ###
# (training epsilon smaller than the half of the smallest distance to different classes)
# Please note that epsilon is not fixed unlike commonly applied adversarial training and epsilon is dependent on the samples
plt.figure(4)
tmp=x1.reshape([1,-1,2])-x2.reshape([-1,1,2])
tmp_norm = np.linalg.norm(tmp, axis=-1, ord=p_ord)
min_norms=np.concatenate([np.min(tmp_norm,axis=0), np.min(tmp_norm,axis=1)],axis=0)
for i, xi in enumerate(x):
    tmp_norm = np.linalg.norm(U_lin - xi, axis=-1, ord=p_ord)
    if i<5:
        plt.contour(X_lin, Y_lin, tmp_norm.reshape([lin_number, lin_number]), levels=[min_norms[i]/2], colors='r', linewidths=2, linestyles='dashed')
    else:
        plt.contour(X_lin, Y_lin, tmp_norm.reshape([lin_number, lin_number]), levels=[min_norms[i]/2], colors='b', linewidths=2, linestyles='dashed')
plt.contour(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), levels=[0.5], colors='black', linewidths=2, linestyles='dashed')
plt.scatter(x1[:,0], x1[:,1],s=20,c='r')
plt.scatter(x2[:, 0], x2[:, 1],s=20, c='b')
plt.xlim(xy_lim)
plt.ylim(xy_lim)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
if p_ord == np.inf:
    plt.savefig('Proper_adversarial_linf.png', bbox_inches='tight', dpi=200)
else:
    plt.savefig('Proper_adversarial_l{}.png'.format(str(p_ord)), bbox_inches='tight', dpi=200)
plt.show()