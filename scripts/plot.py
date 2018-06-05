import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class Plotter:
    def __init__(self, x_label=None, y_label=None):
        """
        :param x_label: is the label for the x axis
        :param y_label: is the label for the y axis
        """
        # data[0] is the x axis
        # data[1] is the y axis
        # data[2] is the figure index
        # data[3] is the label
        # data[4] is the boolean for isCurve
        self.__curve_data = []
        self.__hist_data = []
        self.__x_label = x_label or ''
        self.__y_label = y_label or ''
        self.__figure = 1
        self.__colors = ['dodgerblue', 'darkorange', 'darkgreen', 'purple']
        self.__color_counter = 0
        pass

    def add_curve(self, x_points, y_points, curve_name=None, new_figure=False):
        """
        Add a new curve to the plots
        :param x_points: is the set of x data points
        :param y_points: is the set of y data points
        :param curve_name: is the name to give to the label
        :param new_figure: is true if you want to plot the curve in a separate figure
        :return: None
        """
        assert len(x_points) == len(y_points)
        if new_figure:
            self.__figure += 1
        self.__curve_data.append([x_points, y_points, self.__figure, curve_name, True])

    def add_histogram(self, x_point, y_point, hist_name=None):
        """
        Add a new histogram in a new figure
        :param x_point: is a list of labels, they can be integers or strings
        :param y_point: is a list of quantities mapped on the labels
        :param hist_name: never used for now
        :return: None
        """
        assert len(x_point) == len(y_point)
        self.__figure += 1
        self.__hist_data.append([x_point, y_point, self.__figure, hist_name, False])

    def add_from_csv(self, path):
        pass

    def show(self, separate=True):
        """
        Display all the graph added to the plotter
        :param separate: choose if separate, inside the same figure the curves,
                         if True it will generate a sub graph foreach curve in the figure
                         otherwise it will plot all the curves in the same cartesian axes
        :return: None
        """
        self.__show_curves(separate)
        self.__show_hists()
        for i in range(self.__figure):
            plt.figure(i + 1)
            plt.ylabel(self.__y_label)
            plt.xlabel(self.__x_label)
        plt.show()

    def __next_color(self):
        c = self.__colors[self.__color_counter]
        if self.__color_counter + 1 < len(self.__colors):
            self.__color_counter = self.__color_counter + 1
        else:
            self.__color_counter = 0
        return c

    def __show_curves(self, separate):
        for fig in range(1, self.__figure + 1):
            plt.figure(fig)
            tmp = filter(lambda x: (x[2] == fig), self.__curve_data)
            if separate:
                i = 1
                for curve in tmp:
                    plt.subplot(len(self.__curve_data), 1, i)
                    line, = plt.plot(curve[0], curve[1], label=curve[3] or '')
                    plt.legend(handles=[line], loc=1)
                    i += 1
            else:
                plt.subplot(111)
                i = 0
                for curve in tmp:
                    line, = plt.plot(curve[0], curve[1], label=curve[3] or '')
                    legend = plt.legend(handles=[line], loc=i)
                    plt.gca().add_artist(legend)
                    i += 1

    def __show_hists(self):
        for hist in self.__hist_data:
            data = self.__get_hist_values(hist[0], hist[1])
            counts = Counter(data)

            labels, values = zip(*counts.items())
            labels = np.array(labels)
            values = np.array(values)
            indexes = np.arange(len(labels))

            plt.figure(hist[2], figsize=(plt.rcParams['figure.figsize'][0] / 2, plt.rcParams['figure.figsize'][1]))
            plt.subplot(111)
            bar_width = 0

            plt.bar(indexes, values, color=[self.__next_color() for _ in range(len(indexes))])
            plt.xticks(indexes + bar_width, labels)

    @staticmethod
    def __get_hist_values(x, y):
        assert len(x) == len(y)
        tmp = []
        for i in range(len(y)):
            for j in range(y[i]):
                tmp.append(x[i])

        print(np.shape(tmp))
        print(np.shape(np.random.randn(100)))
        return tmp


if __name__ == '__main__':
    graph = Plotter('label_x', 'label_y')
    graph.add_curve([1, 2, 3], [1, 2, 3], curve_name='c1')
    graph.add_curve([1, 2, 3], [2, 4, 6], curve_name='c2')
    graph.add_curve([1, 2, 3, 4], [1, 4, 9, 16], curve_name='quadratic', new_figure=True)
    graph.add_histogram(['ciao', 'b'], [4, 6])
    graph.show(False)
    pass
