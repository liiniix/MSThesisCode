import matplotlib.pyplot as plt

def multilineplot(data, filename):
    '''
    data:
        data1_y: [],
        data2_y: [],
        ...
        x:       []
    '''
    for (key, value) in data.items():
        if key == 'x':
            continue

        plt.plot(data['x'], value, label=key)

    plt.legend()
    plt.savefig(f"{filename}.png")
    plt.clf()


def showProposedVsOther(data1, data2, data3):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    for (key, value) in data1.items():
        if key == 'x':
            continue

        ax1.plot(data1['x'], value, label=key)

    
    for (key, value) in data2.items():
        if key == 'x':
            continue

        ax2.plot(data2['x'], value, label=key)


    for (key, value) in data3.items():
        if key == 'x':
            continue

        ax3.plot(data3['x'], value, label=key)

    plt.legend()
    plt.savefig("showProposedVsOther.png")
    plt.clf()