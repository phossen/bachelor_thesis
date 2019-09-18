import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from wcds.wcds import WCDS
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import matplotlib
import logging
matplotlib.use('TkAgg')


class DiscrVisual(Frame):
    """
    This application can be used to visualize
    WiSARD's discriminators.
    """

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.discriminators = []
        self.addressing_seed = 123123
        self.random_state = 123123

        # Data set
        self.DATA = make_blobs(
            n_samples=30, n_features=2, centers=[
                (0.5,0.5)], cluster_std=0.05, shuffle=True, random_state=self.random_state)[0]

        # Parameters
        self.OMEGA = IntVar(self.master, 3500)
        self.DELTA = IntVar(self.master, 50)
        self.GAMMA = IntVar(self.master, 50)
        self.BETA = IntVar(self.master, 2)
        self.EPSILON = DoubleVar(self.master, 0.5)
        self.µ = DoubleVar(self.master, 0)
        self.show_points = IntVar(self.master, 0)
        self.show_centroids = IntVar(self.master, 0)

        self.create_widgets()

    def create_widgets(self):
        # General stuff
        self.master.title("Discriminator Visualization")
        pane = PanedWindow(orient=HORIZONTAL)
        pane.pack(fill=BOTH, expand=1)
        left_frame = Frame(pane)
        pane.add(left_frame)
        right_frame = Frame(pane)
        pane.add(right_frame)

        # Combobox and button
        discr_label = Label(right_frame, text="Select discriminator..")
        discr_label.pack()
        self.discriminator_box = ttk.Combobox(
            right_frame, values=self.discriminators, state="readonly")
        self.discriminator_box.pack()
        show_button = Button(
            right_frame,
            text='Update plot',
            width=15,
            command=self.update_plot)
        show_button.pack()

        # Scale definitions
        omega_scale = Scale(
            right_frame,
            from_=1,
            to=10000,
            orient=HORIZONTAL,
            label="Omega",
            variable=self.OMEGA,
            resolution=10)
        omega_scale.pack()
        gamma_scale = Scale(
            right_frame,
            from_=1,
            to=300,
            orient=HORIZONTAL,
            label="Gamma",
            variable=self.GAMMA,
            resolution=1)
        gamma_scale.pack()
        delta_scale = Scale(
            right_frame,
            from_=1,
            to=300,
            orient=HORIZONTAL,
            label="Delta",
            variable=self.DELTA,
            resolution=1)
        delta_scale.pack()
        beta_scale = Scale(
            right_frame,
            from_=1,
            to=100,
            orient=HORIZONTAL,
            label="Beta",
            variable=self.BETA,
            resolution=2)
        beta_scale.pack()
        epsilon_scale = Scale(
            right_frame,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            label="Epsilon",
            variable=self.EPSILON,
            resolution=0.02)
        epsilon_scale.pack()
        mu_scale = Scale(
            right_frame,
            from_=0,
            to=10,
            orient=HORIZONTAL,
            label="µ",
            variable=self.µ,
            resolution=0.1)
        mu_scale.pack()

        # Check boxes
        original_checkbutton = Checkbutton(
            right_frame,
            text="Show originial points",
            variable=self.show_points)
        original_checkbutton.pack()
        centroid_checkbutton = Checkbutton(
            right_frame,
            text="Show centroid",
            variable=self.show_centroids)
        centroid_checkbutton.pack()

        # Plot creation
        self.fig, self.ax = plt.subplots(figsize=(5, 5), dpi=170)
        self.ax.axis("scaled")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        self.canvas.draw()
        self.calculate_clustering(draw_colormap=True)  # Call for first time

        # Buttons
        apply_button = Button(
            right_frame,
            text='Apply',
            width=15,
            command=self.calculate_clustering)
        apply_button.pack()
        save_button = Button(
            right_frame,
            text="Save figure to file",
            width=15,
            command=self.save_figure)
        save_button.pack()

    def save_figure(self, path="discriminator.png"):
        self.fig.savefig(path)

    def update_cbox(self):
        self.discriminator_box["values"] = self.discriminators

    def calculate_clustering(self, draw_colormap=False):
        # Load values
        omega = self.OMEGA.get()
        gamma = self.GAMMA.get()
        delta = self.DELTA.get()
        beta = self.BETA.get()
        epsilon = self.EPSILON.get()
        mu = self.µ.get()

        # Clustering
        self.c_online = WCDS(
            omega, delta, gamma, epsilon, len(
                self.DATA[0]), beta=beta, µ=mu, seed=self.addressing_seed)
        self.assigned_discriminators = []
        for time_, observation in enumerate(self.DATA):
            self.assigned_discriminators.append(
                self.c_online.record(observation, time_)[0])
        self.discriminators = list(self.c_online.discriminators.keys())
        self.update_cbox()
        self.discriminator_box.set(self.discriminators[0])

        # Adjusting sliders to real values of WCDS
        self.GAMMA.set(self.c_online.gamma)
        self.DELTA.set(self.c_online.delta)
        self.BETA.set(self.c_online.beta)

        # Plot
        self.update_plot(draw_colormap=draw_colormap)

    def update_plot(self, step=0.01, draw_colormap=False):
        # Load values
        show_centroid = self.show_centroids.get()
        show_points = self.show_points.get()
        discr_id = int(self.discriminator_box.get())

        # Plotting
        self.ax.clear()
        colormap = cm.get_cmap("viridis")  # Adjust for other colors https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        points = []
        mu = self.µ.get()

        for i in np.arange(0, 1, step):
            for j in np.arange(0, 1, step):
                matching_rate = self.c_online.discriminators[discr_id].matching(
                    self.c_online.addressing((i, j)), mu)
                points.append(((i, j), matching_rate))
        img = self.ax.scatter([point[0][0] for point in points],
                              [point[0][1] for point in points],
                              marker="s",
                              s=1.7, cmap=colormap, c=[point[1] for point in points], vmin=0, vmax=1)
         # Show original points
        if show_points:
            self.ax.scatter([self.DATA[i][0] for i in range(len(self.DATA)) if self.assigned_discriminators[i] == discr_id],
                            [self.DATA[i][1] for i in range(
                                len(self.DATA)) if self.assigned_discriminators[i] == discr_id],
                            marker="o", s=2, color="red")
        # Show centroids
        if show_centroid:
            centroid = self.c_online.centroid(
                self.c_online.discriminators[discr_id])
            self.ax.scatter(centroid[0], centroid[1],
                            marker="X",
                            s=2.5,
                            color="orange")
        if draw_colormap:
            self.fig.colorbar(img, label="Matching")
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = Tk()
    app = DiscrVisual(master=root)
    app.mainloop()
