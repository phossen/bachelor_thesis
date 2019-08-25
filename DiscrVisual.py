import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from wcds.wcds import WCDS
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')


class DiscrVisual(Frame):
    """
    This application can be used to visualize
    WiSARD's discriminators.
    """

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.discriminators = []

        # Data set
        self.DATA = make_blobs(
            n_samples=30, n_features=2, centers=[
                (0.6, 0.6)], cluster_std=0.05, shuffle=True, random_state=None)[0]

        # Parameters
        self.OMEGA = IntVar(self.master, 2000)
        self.DELTA = IntVar(self.master, 50)
        self.GAMMA = IntVar(self.master, 50)
        self.EPSILON = DoubleVar(self.master, 0.5)
        self.µ = DoubleVar(self.master, 0)
        self.show_points = IntVar(self.master, 0)

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
        self.discriminator_box = ttk.Combobox(
            right_frame, values=self.discriminators, state="readonly")
        self.discriminator_box.pack()
        self.show_button = Button(
            right_frame,
            text='Show',
            width=15,
            command=self.update_plot)
        self.show_button.pack()

        # Scale definitions
        self.omega_scale = Scale(
            right_frame,
            from_=1,
            to=10000,
            orient=HORIZONTAL,
            label="Omega",
            variable=self.OMEGA,
            resolution=10)
        self.omega_scale.pack()
        self.delta_scale = Scale(
            right_frame,
            from_=1,
            to=300,
            orient=HORIZONTAL,
            label="Delta",
            variable=self.DELTA,
            resolution=1)
        self.delta_scale.pack()
        self.gamma_scale = Scale(
            right_frame,
            from_=1,
            to=300,
            orient=HORIZONTAL,
            label="Gamma",
            variable=self.GAMMA,
            resolution=1)
        self.gamma_scale.pack()
        self.epsilon_scale = Scale(
            right_frame,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            label="Epsilon",
            variable=self.EPSILON,
            resolution=0.02)
        self.epsilon_scale.pack()
        self.mu_scale = Scale(
            right_frame,
            from_=0,
            to=10,
            orient=HORIZONTAL,
            label="µ",
            variable=self.µ,
            resolution=0.1)
        self.mu_scale.pack()

        # Check boxes
        self.original_checkbutton = Checkbutton(
            right_frame,
            text="Show originial points",
            variable=self.show_points)
        self.original_checkbutton.pack()

        # Plot creation
        self.fig = plt.Figure(figsize=(5, 5), dpi=150)
        self.ax = self.fig.add_subplot()
        self.ax.axis("scaled")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        self.canvas.draw()
        self.calculate_clustering(draw_colormap=True)  # Call for first time

        # Button
        self.apply_button = Button(
            right_frame,
            text='Apply',
            width=15,
            command=self.calculate_clustering)
        self.apply_button.pack()
        self.save_button = Button(
            right_frame,
            text="Save figure to file",
            width=15,
            command=self.save_figure)
        self.save_button.pack()

    def save_figure(self, path="discriminator.png"):
        self.fig.savefig(path)

    def update_cbox(self):
        self.discriminator_box["values"] = self.discriminators

    def calculate_clustering(self, draw_colormap=False):
        # Load values
        data = self.DATA
        omega = self.OMEGA.get()
        gamma = self.GAMMA.get()
        delta = self.DELTA.get()
        epsilon = self.EPSILON.get()
        mu = self.µ.get()

        # Clustering
        self.c_online = WCDS(omega, delta, gamma, epsilon, len(data[0]), µ=mu)
        self.assigned_discriminators = []
        for time_, observation in enumerate(data):
            self.assigned_discriminators.append(
                self.c_online.record(observation, time_))
        self.discriminators = list(self.c_online.discriminators.keys())
        self.update_cbox()
        self.discriminator_box.set(self.discriminators[0])

        # Adjusting sliders to real values of WCDS
        self.GAMMA.set(self.c_online.gamma)
        self.DELTA.set(self.c_online.delta)

        # Plot
        self.update_plot(draw_colormap)

    def update_plot(self, draw_colormap=False):
        # Load values
        data = self.DATA
        show_points = self.show_points.get()
        discr_id = int(self.discriminator_box.get())

        # Plotting
        self.ax.clear()
        step = 0.01
        colormap = cm.get_cmap("viridis")  # Adjust for other colors
        points = []

        for i in np.arange(0, 1, step):
            for j in np.arange(0, 1, step):
                matching_rate = self.c_online.discriminators[discr_id].matching(
                    self.c_online.addressing((i, j)))
                c = colormap(matching_rate)
                points.append(((i, j), c))
        img = self.ax.scatter([point[0][0] for point in points],
                              [point[0][1] for point in points],
                              marker="s",
                              s=1.5,
                              c=[point[1] for point in points])
        if show_points:
            # Show original points and centroid
            self.ax.scatter([data[i][0] for i in range(len(data)) if self.assigned_discriminators[i] == discr_id], [
                            data[i][1] for i in range(len(data)) if self.assigned_discriminators[i] == discr_id], marker="o", s=2, color="blue")
            self.ax.scatter(
                self.c_online.centroid(
                    self.c_online.discriminators[discr_id])[0],
                self.c_online.centroid(
                    self.c_online.discriminators[discr_id])[1],
                marker="X",
                s=2.5,
                color="red")
        if draw_colormap:
            self.fig.colorbar(img)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = Tk()
    app = DiscrVisual(master=root)
    app.mainloop()
