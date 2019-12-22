"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 3 - Activity prediction for chemical compounds

Authors :
    - Maxime Meurisse
    - Francois Rozet
    - Valentin Vermeylen
"""

#############
# Libraries #
#############

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib.ticker import MaxNLocator


#################
# Configuration #
#################

rc('text', usetex=True)


#############
# Functions #
#############

def line_chart(x, y, x_label, y_label, fig_name):
    # Initialize figure
    plt.figure()

    try:
        # Create figure
        fig, ax = plt.subplots()

        # Add the line
        ax.plot(x, y)

        # Add the maximum point
        y_max = max(y)
        x_max = x[np.argmax(y)]

        ax.plot(x_max, y_max, 'ro')

        # Add axis label
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')

        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(True, color='#EEEEEE')

        # Tick parameters
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Generate and save figure
        fig.tight_layout()
        fig.savefig('{}.pdf'.format(fig_name))
    finally:
        plt.close()


def bar_chart(data, data_labels, x_label, y_label, x_legend, fig_name):
    # Initialize figure
    plt.figure()

    try:
        n_data = len(data)
        bar_width = (1 / n_data) - 0.1

        # Set positions of bars on x axis
        r = list()
        r.append(np.arange(len(x_legend)))

        for i in range(n_data - 1):
            r.append([x + bar_width for x in r[-1]])

        # Create figure
        fig, ax = plt.subplots()

        # Add the bars
        for i, d in enumerate(data):
            ax.bar(r[i], d, bar_width, label=data_labels[i])

        # Add axis label
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Axis styling.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')

        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)

        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)

        # Add legend
        ax.set_xticks(r[0] + (n_data - 1) * (bar_width / 2))
        ax.set_xticklabels(x_legend)

        ax.legend()

        # Generate and save figure
        fig.tight_layout()
        fig.savefig('{}.pdf'.format(fig_name))
    except:
        plt.close()
