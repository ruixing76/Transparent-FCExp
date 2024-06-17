from matplotlib import pyplot as plt
import seaborn as sns


def plot_histogram(data, title, x_value, x_label, y_label):
    plt.figure(figsize=(5, 2))
    if x_value is not None:
        sns.histplot(data=data,x_value=data[x_value], kde=True)
    else:
        sns.histplot(data=data, kde=True)
    plt.title(f'Histogram of {title}')
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)
    plt.show()


if __name__ == '__main__':
    pass