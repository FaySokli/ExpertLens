from deepview.embeddings import init_umap, init_inv_umap
from deepview.fisher_metric import calculate_fisher
from deepview.Selector import SelectFromCollection
from deepview.DeepView import DeepView


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
import os

class DeepViewIR(DeepView):
    """
    DeepViewIR extends DeepView to support Information Retrieval (IR) workflows.

    It introduces the concept of:
    - a query (the main sample of interest)
    - relevant documents (highlighted separately)
    - optional relevance-based visualization logic

    This class inherits everything from DeepView, so it remains backward-compatible.
    """

    def __init__(self, pred_fn, classes, max_samples, batch_size, data_shape, n=5,
                 lam=0.65, resolution=100, cmap='tab10', interactive=True, verbose=True,
                 title='DeepView', data_viz=None, mapper=None, inv_mapper=None, metric="euclidean",
                 disc_dist=True, use_selector=False, class_dict=None, relevant_docs=None, **kwargs):

        super().__init__(
            pred_fn=pred_fn,
            classes=classes,
            max_samples=max_samples,
            batch_size=batch_size,
            data_shape=data_shape,
            n=n,
            lam=lam,
            resolution=resolution,
            cmap=cmap,
            interactive=interactive,
            verbose=verbose,
            title=title,
            data_viz=data_viz,
            mapper=mapper,
            inv_mapper=inv_mapper,
            metric=metric,
            disc_dist=disc_dist,
            use_selector=use_selector,
            class_dict=class_dict,
            relevant_docs=relevant_docs,
            **kwargs
        )

        self.relevant_docs = relevant_docs if relevant_docs is not None else []

    # ---------------------
    # IR-SPECIFIC METHODS
    # ---------------------

    def _init_plots(self):
        '''
        Initialises matplotlib artists and plots.
        '''
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_title(self.title)
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        class_label_display = (self.class_dict if self.class_dict is not None else self.classes)
        for c in range(self.n_classes):
            color = self.cmap(c/(self.n_classes-1))
            plot = self.ax.plot([], [], 'o', label=class_label_display[c],
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        for c in range(self.n_classes):
            color = self.cmap(c/(self.n_classes-1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                fillstyle='none', ms=12, mew=2.5, zorder=1)
            self.sample_plots.append(plot[0])

        # query and relevant document plots
        self.query_plot, = self.ax.plot([], [], 'o',
            markeredgecolor='yellow', mew=2.5, ms=14, zorder=3)

        self.relevant_plots, = self.ax.plot([], [], 'o',
            markeredgecolor='red', mew=2.5, ms=12, zorder=3)

        # set the mouse-event listeners
        if self.use_selector:
            self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
        else:
            self.fig.canvas.mpl_connect('pick_event', self.show_sample)
            self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.set_axis_off()
        self.ax.legend()

    # ---------------------
    # EXTENDED SHOW METHOD
    # ---------------------

    def show(self):
        """
        Extends the DeepView show() method to add IR-specific visualization
        such as highlighting the query and relevant documents differently.
        """
        if not hasattr(self, 'fig'):
            self._init_plots()

        x_min, y_min, x_max, y_max = self._get_plot_measures()

        self.cls_plot.set_data(self.classifier_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
        desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
        self.desc.set_text(desc)

        for c in range(self.n_classes):
            data = self.embedded[self.y_true == c]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.n_classes):
            data = self.embedded[np.logical_and(self.y_pred == c, self.background_at != c)]
            self.sample_plots[self.n_classes + c].set_data(data.transpose())

        # skipping:
        # if os.name == 'posix':
        #     self.fig.canvas.manager.window.raise_()

        if self.use_selector:
            self.selector = SelectFromCollection(self.ax, self.embedded)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # skipping plt.show()

        # add IR-specific visualization
        self._highlight_ir_points()
    
    def _highlight_ir_points(self):
        """IR-specific logic for queries and relevant docs."""
        # relevant documents
        if len(self.relevant_docs) > 0:
            rel_points = self.embedded[self.relevant_docs]
            rel_classes = [int(self.y_pred[i]) for i in self.relevant_docs]
            self.relevant_plots.set_data(rel_points[:,0], rel_points[:,1])
            
            self.relevant_plots.remove()  # remove old line2D
            self.relevant_plots = self.ax.scatter(
                rel_points[:,0], rel_points[:,1],
                c=[self.cmap(c/(self.n_classes-1)) for c in rel_classes],
                edgecolors='red', linewidths=2.0, s=80, zorder=3
            )

        # query
        query_point = self.embedded[0:1]
        query_class = int(self.y_pred[0])
        self.query_plot.set_data(query_point[:,0], query_point[:,1])
        self.query_plot.set_markerfacecolor(self.cmap(query_class/(self.n_classes-1)))
