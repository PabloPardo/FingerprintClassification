__author__ = 'pablo'
import matplotlib.pyplot as plt

def plot_scores(results, name):
    sort_keys = sorted(results.keys())
    S1 = [results[r]['inter'] for r in sort_keys]
    S2 = [results[r]['inner'] for r in sort_keys]
    S3 = [results[r]['w_inner'] for r in sort_keys]
    S4 = [results[r]['iou'] for r in sort_keys]

    plt.figure()
    plt.plot(S1)
    plt.plot(S2)
    plt.plot(S3)
    plt.plot(S4)

    plt.xticks(range(len(S1)), sort_keys, size='small')

    plt.ylabel('Score')
    plt.xlabel('Number of Classes')
    plt.legend(['InterClass Acc.', 'InnerClass Acc.', 'W. InnerClass Acc.', 'Int. over Uni.'])

    plt.savefig(name, bbox_inches='tight')