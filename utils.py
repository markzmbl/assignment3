import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from KMeans import KMeans
from datetime import datetime
from datetime import timedelta
now = datetime.now


def convergence_plot(centroids_history):
    fig, ax = plt.subplots()
    centroid_distances = np.ndarray(
        (len(centroids_history)-1, len(centroids_history[0]))
    )
    for i, iteration in enumerate(centroids_history[:-1]):
        next_iteration = centroids_history[i+1]
        for j in range(len(iteration)):
            centroid_distances[i, j] = np.linalg.norm(
                iteration[j] - next_iteration[j]
            )

    for history in centroid_distances.T:
        ax.plot(history)
    
    return fig, ax

def benchmark(X, y, k, runs, max_iter, atol, method, title, **kwargs):
    acc_seconds = timedelta()
    acc_nmi = 0
    for i in range(runs):
        model = KMeans(k, max_iter=max_iter, atol=atol, method=method, **kwargs)
        t0 = now()
        model.fit(X)
        t0 = now() - t0
        acc_seconds += t0
        iterations = len(model.centroids_history)
        nmi = normalized_mutual_info_score(y, model.predict(X), average_method='arithmetic')
        acc_nmi += nmi
        fig, ax = convergence_plot(model.centroids_history)
        ax.set_title(
            f'{title}\n'
            f'Run: {i}\n'
            f'Runtime: {t0.seconds} seconds\n'
            f'NMI: {round(nmi, 2)}\n'
            f'Iterations: {iterations}\n'
            f'Distance Computations: {model.distance_computations:.2e}'
    )
    return acc_nmi/runs, acc_seconds/runs

def report_plot(report):
    for metric in 'nmi', 'runtime':

        fig, ax = plt.subplots()
        plt.xticks(rotation=90)

        for key, val in report.items():
            ax.set_xlabel('method')
            if metric == 'nmi':
                measure = val[0]
                ax.set_title('Normalized Mutual Info Score')
                ax.set_ylabel('nmi')
            elif metric == 'runtime':
                measure = val[1]
                measure = measure.seconds + measure.microseconds / 1_000_000
                ax.set_title('Average runtime')
                ax.set_ylabel('seconds')
            if key.startswith('Lyod'):
                ax.bar(key, measure, color='blue')
            elif key.startswith('Local'):
                ax.bar(key.split(' - ')[1], measure, color='green')
            elif key.startswith('Core'):
                ax.bar(key.split(' - ')[1], measure, color='orange')