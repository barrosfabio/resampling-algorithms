from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )

    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(label + '.png')

# define dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=1, class_sep=0.28)

plot_2d_space(X, y, 'Imbalanced dataset - No Resampling')

random = RandomOverSampler()
smote = SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=0, n_jobs=4)
borderline = BorderlineSMOTE(k_neighbors=5, sampling_strategy='auto', random_state=0, n_jobs=4)
smote_enn = SMOTEENN(sampling_strategy='auto', random_state=0, n_jobs=4)
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=0, n_jobs=4)
adasyn = ADASYN(n_neighbors=5,sampling_strategy='auto', random_state=0, n_jobs=4)

X_random, y_random = random.fit_resample(X, y)
X_smote, y_smote = smote.fit_resample(X, y)
X_borderline, y_borderline = borderline.fit_resample(X, y)
X_enn, y_enn = smote_enn.fit_resample(X, y)
X_tomek, y_tomek = smote_tomek.fit_resample(X, y)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

plot_2d_space(X_smote, y_smote, 'Imbalanced dataset - Random Oversampling')
plot_2d_space(X_smote, y_smote, 'Imbalanced dataset - SMOTE')
plot_2d_space(X_borderline, y_borderline, 'Imbalanced dataset - Borderline SMOTE')
plot_2d_space(X_enn, y_enn, 'Imbalanced dataset - SMOTE ENN')
plot_2d_space(X_tomek, y_tomek, 'Imbalanced dataset - SMOTE Tomek')
plot_2d_space(X_adasyn, y_adasyn, 'Imbalanced dataset - ADASYN')

