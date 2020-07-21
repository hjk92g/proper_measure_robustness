import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier

#Use ensemble for estimating optimally robust classifiers

np.random.seed(1)

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
'''np.random.seed(0)
x1=10*np.random.normal(size=[100,1])
x1=np.concatenate([np.cos(x1),1+np.sin(x1)],axis=1)
x2=2*np.random.normal(size=[100,2])
x2[:,1]=0.0
xy_lim=[-4,4]'''


lin_number=100
x_lin = np.linspace(xy_lim[0], xy_lim[1], lin_number)
y_lin = np.linspace(xy_lim[0], xy_lim[1], lin_number)
X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
U_lin = np.concatenate((X_lin.reshape([-1,1]),Y_lin.reshape([-1,1])),axis=1)

div=4

sigma=(1/div)*(np.arange(div)+1)
cm = plt.get_cmap('jet')
print('Sigma values:',sigma)

swc_gradual = 0 #Switch for plotting gradual 1-NN or regular 1-NN

[n1, n2]=[len(x1), len(x2)]
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

plt.figure(div+1)
plt.contourf(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), 20, cmap='jet')
plt.contour(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), levels=[0.5], colors='black', linewidths=2, linestyles='dashed')
plt.scatter(x1[:,0], x1[:,1],s=20,c='r')
plt.scatter(x2[:, 0], x2[:, 1],s=20, c='b')
plt.xlim(xy_lim)
plt.ylim(xy_lim)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

if swc_gradual==1:
	plt.savefig('Classifier_gradual_l1.png', bbox_inches='tight', dpi=200)
	np.save('gradual_l1_prob_result', prob_result)
else:
	plt.savefig('Classifier_l1.png', bbox_inches='tight', dpi=200)
	np.save('l1_prob_result',prob_result)

n_noise= 5000 #Number of noise (Smaller values are preferable for large data size or to get results faster)
t1=time.time()
for i in range(div):
	plt.figure(i+1)
	t2 = time.time()
	print(i+1, 'Time:', t2-t1)
	prob_result = np.zeros(len(U_lin))
	if swc_gradual==1:
		### Ensembel of gradual 1-NNs on noisy data ###
		for j in range(n_noise):
			x1_noise = x1 + noise_gen(sigma[i],n1,ord=p_ord)
			x2_noise = x2 + noise_gen(sigma[i],n2,ord=p_ord)

			diff1 = U_lin.reshape([1, len(U_lin), -1]) - x1_noise.reshape([n1, 1, -1])
			diff2 = U_lin.reshape([1, len(U_lin), -1]) - x2_noise.reshape([n2, 1, -1])
			diff1_norm = np.linalg.norm(diff1, axis=-1, ord=p_ord)
			diff2_norm = np.linalg.norm(diff2, axis=-1, ord=p_ord)
			nn1_norm = np.min(diff1_norm, axis=0)  # Nearest neighbor distance to class 1
			nn2_norm = np.min(diff2_norm, axis=0)  # Nearest neighbor distance to class 2
			prob_result += (1 / nn1_norm) / ((1 / nn1_norm) + (1 / nn2_norm))  # Use 1/x for normalization in gradual 1-NN
	else:
		### Ensemble of standard 1-NNs on noisy data ###
		for j in range(n_noise):
			x1_noise = x1 + noise_gen(sigma[i],n1,ord=p_ord)
			x2_noise = x2 + noise_gen(sigma[i],n2,ord=p_ord)

			neigh = KNeighborsClassifier(n_neighbors=1, p=p_ord)
			neigh.fit(np.concatenate([x1_noise, x2_noise], 0), [1]*n1+[0]*n2)
			tmp_pred = neigh.predict_proba(U_lin)
			tmp_pred = tmp_pred[:, 1]
			tmp_pred[np.where(tmp_pred < 0.5)[0]] = 0
			tmp_pred[np.where(tmp_pred > 0.5)[0]] = 1

			neigh = KNeighborsClassifier(n_neighbors=1, p=p_ord)
			neigh.fit(np.concatenate([x2_noise, x1_noise], 0), [0] * n2 + [1] * n1) # Flipped
			tmp_pred2 = neigh.predict_proba(U_lin)
			tmp_pred2 = tmp_pred2[:, 1]
			tmp_pred2[np.where(tmp_pred2 < 0.5)[0]] = 0
			tmp_pred2[np.where(tmp_pred2 > 0.5)[0]] = 1
			prob_result += (tmp_pred+tmp_pred2)/2

	prob_result = prob_result / n_noise

	plt.contourf(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), 20, cmap='jet')
	plt.contour(X_lin, Y_lin, prob_result.reshape([lin_number, lin_number]), levels=[0.5], colors='black', linewidths=2, linestyles='dashed')
	plt.scatter(x1[:,0], x1[:,1],s=20,c='r')
	plt.scatter(x2[:, 0], x2[:, 1],s=20, c='b')
	plt.xlim(xy_lim)
	plt.ylim(xy_lim)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	if swc_gradual == 1:
		plt.savefig('Classifier_gradual_l1_{}.png'.format(str(i+1)), bbox_inches='tight', dpi=200)
		np.save('gradual_l1_prob_result_{}'.format(str(i + 1)), prob_result)
	else:
		plt.savefig('Classifier_l1_{}.png'.format(str(i + 1)), bbox_inches='tight', dpi=200)
		np.save('l1_prob_result_{}'.format(str(i + 1)), prob_result)

plt.show()

