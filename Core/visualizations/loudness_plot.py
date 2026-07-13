import matplotlib.pyplot as plt
import numpy as np


def plot_loudness(loudness, time_axis):

    

    plt.figure(figsize=(12, 4))

    plt.plot(time_axis, loudness)

    plt.title("Zwicker Loudness")

    plt.xlabel("Time (s)")

    plt.ylabel("Loudness (Sone)")

    plt.grid(True)

    plt.show(block=False)

    plt.pause(10)