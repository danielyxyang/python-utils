__all__ = ["PlotSaver", "DynamicPlotter"]

import os
import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import Divider, Size

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display


class PlotSaver():
    r"""Class providing support for saving PDF plots in LaTeX style.

    Examples:
        ```
        PlotSaver.setup(interactive=True, is_colab=False)
        PlotSaver.configure(
            basewidth=BASEWIDTH,
            latex=True, latex_preamble="\n".join([
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath}",
            ]),
            save_format="pdf",
        )

        fig, axis = PlotSaver.create()
        # PLOTTING CODE
        PlotSaver.finish((fig, "FILENAME"), relsize=0.5, save=True)
        ```
    """
    interactive = False
    is_colab = False
    
    basewidth = 6 # inches
    save_dir = "."
    save_format = "png"
    save_always = False

    # mapping for shorter keys usable as keyword arguments
    _FONTSIZE_KEYS = {
        "base":   "font.size",
        "title":  "axes.titlesize",
        "label":  "axes.labelsize",
        "xtick":  "xtick.labelsize",
        "ytick":  "ytick.labelsize",
        "legend": "legend.fontsize",
    }

    # SETUP FUNCTIONS

    @staticmethod
    def setup(interactive=None, is_colab=None):
        """Setup environment for PlotSaver.

        Args:
            interactive (bool, optional): Flag whether the plots should be
                interactive or not. This only works in an interactive IPython
                environment. Defaults to None.
            is_colab (bool, optional): Flag whether the PlotSaver is used in
                Google Colab or not. This is required for proper functioning of
                interactive plots in Colab. Defaults to None.
        """
        if interactive is not None: PlotSaver.interactive = interactive
        if is_colab is not None:    PlotSaver.is_colab = is_colab
        
        # enable interactive plots in IPython environment
        ip = get_ipython()
        if ip is not None:
            # setup matplotlib backend with IPython magic
            if PlotSaver.interactive:
                ip.run_line_magic("matplotlib", "ipympl")
            else:
                ip.run_line_magic("matplotlib", "inline")

            # enable interactive plots on Colab
            # https://matplotlib.org/ipympl/installing.html#google-colab
            if PlotSaver.interactive and PlotSaver.is_colab:
                from google.colab import output # pyright: ignore[reportMissingImports]
                output.enable_custom_widget_manager()

        # prevent figures to be displayed without calling plt.show() or display()
        plt.ioff()

    @staticmethod
    def configure(
        basewidth=None,
        # parameters for rcParams
        style=None, fontsize=None,
        latex=None, latex_preamble="",
        rcparams=None,
        # parameters for save
        save_dir=None, save_format=None, save_always=None,
    ):
        """Configure default behavior of PlotSaver.

        The default style of the plots can be configured with Matplotlib's
        rcParams and style sheets. The default rcParams can be found here [1]
        and a list of different built-in style sheets here [2].
        
        Args:
            basewidth (float, optional): The basewidth of the area in which the
                plots are used in inches. This can be the width of the output
                area or the textwidth of your LaTeX document. Defaults to None.
            style (str, dict, Path or list, optional): The style specification
                for Matplotlib. Possible specifications are "original",
                "seaborn" or any of the specifications accepted by
                `mpl.style.use`. Defaults to None.
            fontsize (float, str or dict, optional): The default fontsize. A
                scalar defines the font size for all text and a dict defines the
                font sizes according to _FONTSIZE_KEYS. A default font size can
                be specified in the dict under "default". Defaults to None.
            latex (bool, optional): Flag whether the plots should be rendered
                with LaTeX or not. Defaults to None.
            latex_preamble (str, optional): The LaTeX preamble used to render
                the plots. Defaults to "".
            rcparams (dict, optional): The dictionary with additional rcParams
                for Matplotlib. Defaults to None.
            save_dir (str, optional): The path to the directory in which the
                plots should be saved. Defaults to None.
            save_format (str, optional): The format in which the plot should be
                saved. Formats include "png", "pdf", "pgf" and "tikz". Saving as
                "tikz" requires the package `tikzplotlib`. Defaults to None.
            save_always (bool, optional): Flag whether plots should be always
                saved or not. Defaults to None.
        
        References:
            [1] https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
            [2] https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        """
        if basewidth is not None:   PlotSaver.basewidth = basewidth
        if save_dir is not None:    PlotSaver.save_dir = save_dir
        if save_format is not None: PlotSaver.save_format = save_format
        if save_always is not None: PlotSaver.save_always = save_always

        # update rcParams with style
        if style is not None:
            if style == "original":
                # use original rc file loaded by Matplotlib (rcParamsOrig)
                mpl.rc_file_defaults()
            elif style == "seaborn":
                # use original seaborn theme
                import seaborn as sns # pyright: ignore[reportMissingImports]
                sns.set_theme()
            else:
                # use given style specification
                mpl.style.use(style)

        # update rcParams with fontsize
        if fontsize is not None: # default font size: 10pt
            if np.isscalar(fontsize):
                plt.rcParams.update({
                    key: fontsize
                    for _, key in PlotSaver._FONTSIZE_KEYS.items()
                })
            elif isinstance(fontsize, dict):
                default = fontsize.get("default", None)
                plt.rcParams.update({
                    key: fontsize.get(key_short, default)
                    for key_short, key in PlotSaver._FONTSIZE_KEYS.items()
                    if key_short in fontsize or default is not None
                })

        # update rcParams to enable or disable LaTeX rendering
        if latex is not None:
            if latex:
                plt.rcParams.update({
                    # settings for font
                    "font.family": "serif",
                    # settings for rendering text with latex
                    "text.usetex": True,
                    "text.latex.preamble": latex_preamble, # for legend entries
                    # settings for rendering plot with latex
                    "pgf.texsystem": "pdflatex",
                    "pgf.rcfonts": False, # disable font setup from rc parameters
                    "pgf.preamble": latex_preamble,
                })
            else:
                plt.rcParams.update({"text.usetex": False})
        
        # update rcParams with custom parameters
        if rcparams is not None:
            plt.rcParams.update(rcparams)

    @staticmethod
    @contextlib.contextmanager
    def config(**kwargs):
        """Configure default behavior of PlotSaver with context manager."""
        try:
            # save parameters of PlotSaver
            prev = (
                PlotSaver.basewidth,
                PlotSaver.save_dir,
                PlotSaver.save_format,
                PlotSaver.save_always,
            )
            # save and restore rcParams of Matplotlib
            with plt.rc_context():
                # change configuration
                PlotSaver.configure(**kwargs)
                yield
        finally:
            # restore parameters of PlotSaver
            (
                PlotSaver.basewidth,
                PlotSaver.save_dir,
                PlotSaver.save_format,
                PlotSaver.save_always,
            ) = prev
    
    # PLOTTING FUNCTIONS

    @staticmethod
    def display_html_hack():
        """Display CSS hack to show toolbar of interactive plots in Colab."""
        if PlotSaver.interactive and PlotSaver.is_colab:
            html_hack = widgets.HTML("<style> .jupyter-matplotlib-figure { position: relative; } </style>")
            display(html_hack)

    @staticmethod
    def create(**kwargs):
        """Create a new plot and set the style.

        Args:
            **kwargs: Keyword arguments for `set_style` function.
        Returns:
            Tuple of Figure and Axes instance.
        """        
        # create figure
        fig, axis = plt.subplots(constrained_layout=True)
        PlotSaver.set_style(axis, **kwargs)

        # configure style for interactive plots
        if PlotSaver.interactive:
            fig.canvas.toolbar_position = "top"
            fig.canvas.header_visible = False

        return fig, axis

    @staticmethod
    def finish(
        plots,
        # parameters for figure size
        figsize=None, relsize=None, ratio=None,
        consistent_size=True,
        tight_layout=True,
        # parameters for displaying figures
        show=True, ncols=4,
        # parameters for saving figures
        save=False, save_format=None,
        # parameters for final adjustments
        **kwargs,
    ):
        """Set the final style and finish the plot by displaying and saving it.

        Args:
            plots (tuple (Figure, str) or list): (figure, filename) or a list of
                such tuples.
            figsize (tuple (float, float), optional): (width, height) Width and
                height in inches. Defaults to None.
            relsize (float, optional): Width of figure relative to specified
                textwidth. Defaults to None.
            ratio (float, optional): Ratio height/width of axes, not of figure.
                Defaults to None.
            consistent_size (bool, optional): Flag whether to consistently size
                the Axes by unifying the side padding. Defaults to True.
            tight_layout (bool, optional): Flag whether to change layout engine
                to tight_layout. Defaults to True.
            show (bool, optional): Flag whether to display the plot. Defaults to
                True.
            ncols (int, optional): Number of columns for displaying the list of
                plots. Defaults to 4.
            save (bool, optional): Flag whether to save the plot. Defaults to
                False.
            save_format (str, optional): The format in which the plot should be
                saved. Formats include "png", "pdf", "pgf" and "tikz". Saving as
                "tikz" requires the package `tikzplotlib`. Defaults to None.
            **kwargs: Keyword arguments for `set_style` function.
        """
        if not isinstance(plots, list):
            plots = [plots]
        if len(plots) == 0:
            return
        # set ratio            
        if ratio is None:
            ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean
        # set figsize
        if figsize is not None:
            ratio = figsize[1] / figsize[0]
        else:
            if relsize is not None:
                width = relsize * PlotSaver.basewidth
            else:
                width = PlotSaver.basewidth
            figsize = (width, width * ratio)
        
        # make final adjustments
        for fig, _ in plots:
            if fig is None:
                continue
            # set style
            PlotSaver.set_style(fig.gca(), **kwargs)
            # set sizing
            fig.set_size_inches(figsize)
            if tight_layout:
                fig.tight_layout(pad=0.25)
        
        # ensure consistent sizing (e.g. for side-by-side plots)
        # https://stackoverflow.com/a/52052892
        if consistent_size:
            def get_padding(fig):
                # return required side-space for title, labels, ticks, etc.
                l = fig.subplotpars.left * fig.get_figwidth()
                r = (1 - fig.subplotpars.right) * fig.get_figwidth()
                b = fig.subplotpars.bottom * fig.get_figheight()
                t = (1 - fig.subplotpars.top) * fig.get_figheight()
                return np.array([l, r, b, t])
            # compute max required fixed side-space
            l_max, r_max, b_max, t_max = np.max([get_padding(fig) for fig, _ in plots if fig is not None], axis=0)
            horiz = [Size.Fixed(l_max), Size.Scaled(1), Size.Fixed(r_max)]
            verti = [Size.Fixed(b_max), Size.Scaled(ratio), Size.Fixed(t_max)]
            # compute new height to adapt ratio to ratio of axes instead of figure
            fig_width = figsize[0]
            axes_width = fig_width - l_max - r_max
            axes_height = axes_width * ratio
            fig_height = axes_height + t_max + b_max
            for fig, _ in plots:
                if fig is None:
                    continue
                # place axis in divider
                divider = Divider(fig, (0, 0, 1, 1), horiz, verti, aspect=True)
                fig.gca().set_axes_locator(divider.new_locator(nx=1, ny=1))
                # set new height
                fig.set_figheight(fig_height)

        # show figures
        if show or PlotSaver.save_always:
            PlotSaver.display_html_hack()
            grid = widgets.GridspecLayout(n_rows=int(np.ceil(len(plots)/ncols)), n_columns=ncols, width="{}in".format((figsize[0]+0.5)*ncols))
            for i, (fig, name) in enumerate(plots):
                if fig is None:
                    continue
                out = widgets.Output(layout=dict(overflow="auto"))
                with out:
                    if PlotSaver.interactive and not PlotSaver.save_always:
                        display(fig.canvas)
                    else:
                        display(fig)
                    display(widgets.Label("{:3}: {}".format(fig.number, name), layout=dict(overflow="auto"), style=dict(font_family="monospace", font_size="10pt")))
                grid[i // ncols, i % ncols] = out
            display(grid)

        # save figures
        if save or PlotSaver.save_always:
            if save_format is None or PlotSaver.save_always:
                save_format = PlotSaver.save_format
            if save_format in ["png", "tikz", "pgf", "pdf"]:
                for fig, name in plots:
                    if fig is None:
                        continue
                    # setup output path
                    filepath = os.path.join(PlotSaver.save_dir, save_format, name)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    # compute width and height annotations
                    width_in, height_in = fig.get_size_inches()
                    width = "{:.2f}in".format(width_in) if relsize is None else r"{}\textwidth".format(relsize)
                    height = "{:.2f}in".format(height_in) if relsize is None else r"{}\textwidth".format(relsize * height_in/width_in)
                    
                    # save figure to file
                    if save_format == "png":
                        filepath += ".png"
                        fig.savefig(filepath, dpi=150)
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print(r"\includegraphics[width=%s]{%s}" % (width, name))
                    elif save_format == "pdf":
                        filepath += ".pdf"
                        fig.savefig(filepath, backend="pgf")
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print(r"\includegraphics[width=%s]{%s.pdf}" % (width, name))
                    elif save_format == "pgf":
                        filepath += ".pgf"
                        fig.savefig(filepath, backend="pgf")
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print(r"\resizebox{%s}{!}{\input{%s.pgf}}" % (width, name))
                    elif save_format == "tikz":
                        import tikzplotlib as tikz
                        filepath += ".tex"
                        tikz.save(filepath, axis_width="r\\tikzwidth", axis_height="\\tikzheight", wrap=False)
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print("\n".join([
                            r"\begin{tikzpicture}",
                            r"    \def\tikzwidth{%s}",
                            r"    \def\tikzheight{%s}",
                            r"    \input{%s}",
                            r"\end{tikzpicture}",
                        ]) % (width, height, name))
            else:
                print("WARNING: not supported to save plot in {} format".format(save_format))

    @staticmethod
    def set_style(
        axis,
        title=None,
        xlabel=None, ylabel=None,
        xlim=None, ylim=None,
        xmargin=None, ymargin=None,
        xticks=None,
        yticks=None,
        centeraxes=False,
        legend=False,
    ):
        """Set all relevant style for an axis with a single function call.

        Args:
            axis (Axes): Instance of matplotlib Axes.
            title (str or dict, optional): Title of Axes or dict with arguments
                for `set_title`. Defaults to None.
            xlabel (str or dict, optional): Label for x-axis of Axes or dict
                with arguments for `set_xlabel`. Defaults to None.
            ylabel (str or dict, optional): Label for y-axis of Axes or dict
                with arguments for `set_ylabel`. Defaults to None.
            xlim (tuple, optional): Left and right xlims. Defaults to None.
            ylim (tuple, optional): Bottom and top ylims. Defaults to None.
            xmargin (float, optional): Relative margin to the left and right.
                This is ignored if xlim is specified. Defaults to None.
            ymargin (float, optional): Relative margin to the bottom and top.
                This is ignored if ylim is specified. Defaults to None.
            xticks (list or dict, optional): List of xticks or dict(ticks=X,
                labels=X) or dict(locator=X, formatter=X). Defaults to None.
            yticks (list or dict, optional): List of yticks or dict(ticks=X,
                labels=X) or dict(locator=X, formatter=X). Defaults to None.
            centeraxes (bool, optional): Flag whether to center the axes.
                Defaults to False.
            legend (bool or dict, optional): Flag whether to show the legend
                with default options or dict with kwargs for legend() and
                additional "order" keyword for specifying order of artists by
                position. Defaults to False.
        """        
        # set text
        if title is not None:
            if isinstance(title, str):    axis.set_title(title)
            elif isinstance(title, dict): axis.set_title(**title)
        if xlabel is not None:
            if isinstance(xlabel, str):    axis.set_xlabel(xlabel)
            elif isinstance(xlabel, dict): axis.set_xlabel(**xlabel)
        if ylabel is not None:
            if isinstance(ylabel, str):    axis.set_ylabel(ylabel)
            elif isinstance(ylabel, dict): axis.set_ylabel(**ylabel)
        # set ticks (before limits)
        if xticks is not None:
            if isinstance(xticks, list):
                axis.set_xticks(ticks=xticks)
            elif isinstance(xticks, dict) and "ticks" in xticks and "labels" in xticks:
                axis.set_xticks(ticks=xticks["ticks"], labels=xticks["labels"])
            elif isinstance(xticks, dict) and "locator" in xticks and "formatter" in xticks:
                axis.xaxis.set_major_locator(xticks["locator"])
                axis.xaxis.set_major_formatter(xticks["formatter"])
        if yticks is not None:
            if isinstance(yticks, list):
                axis.set_yticks(ticks=yticks)
            elif isinstance(yticks, dict) and "ticks" in yticks and "labels" in yticks:
                axis.set_yticks(ticks=yticks["ticks"], labels=yticks["labels"])
            elif isinstance(yticks, dict) and "locator" in yticks and "formatter" in yticks:
                axis.yaxis.set_major_locator(yticks["locator"])
                axis.yaxis.set_major_formatter(yticks["formatter"])
        # set limits
        if xlim is not None:      axis.set_xlim(xlim)
        elif xmargin is not None: axis.set_xmargin(xmargin)
        if ylim is not None:      axis.set_ylim(ylim)
        elif ymargin is not None: axis.set_ymargin(ymargin)
        # center axes
        if centeraxes:
            # center left and bottom axes
            axis.spines["left"].set_position("zero")
            axis.spines["bottom"].set_position("zero")
            # hide upper and right axes
            axis.spines["right"].set_color("none")
            axis.spines["top"].set_color("none")
            # add arrow tips to left and bottom axes
            axis.plot(1, 0, ">", color="black", transform=axis.get_yaxis_transform(), clip_on=False)
            axis.plot(0, 1, "^", color="black", transform=axis.get_xaxis_transform(), clip_on=False)
        # set legend
        if legend:
            if isinstance(legend, dict):
                if "order" in legend:
                    order = legend.pop("order")
                    handles, labels = axis.get_legend_handles_labels()
                    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
                    legend["handles"] = handles
                    legend["labels"] = labels
                axis.legend(**legend)
            else:
                axis.legend()

    # UTILITY FUNCTIONS
    
    @staticmethod
    def print_lim(fig_index, prec=2):
        """Print the current view limits of the figure."""
        fig = plt.figure(fig_index)
        print("({:.{prec}f}, {:.{prec}f})".format(*fig.gca().get_xlim(), prec=prec))
        print("({:.{prec}f}, {:.{prec}f})".format(*fig.gca().get_ylim(), prec=prec))

    @staticmethod
    def transfer_lim(fig_index_from, fig_index_to):
        """Transfer the view limits from one figure to another."""
        fig_from = plt.figure(fig_index_from)
        fig_to = plt.figure(fig_index_to)
        fig_to.gca().set_xlim(fig_from.gca().get_xlim())
        fig_to.gca().set_ylim(fig_from.gca().get_ylim())
        fig_to.canvas.draw()
        fig_to.canvas.flush_events()


class DynamicPlotter():
    """Class providing support for dynamically updating plots."""
    interactive = False

    @staticmethod
    def set_interactive(interactive=True):
        """Enable or disable interactive plots based on ipyml backend."""
        DynamicPlotter.interactive = interactive

    def __init__(self):
        self.fig = None
        self.axis = None # currently active axis
        self.artists = {}
        self.__displayed = False
    
    def create(self):
        """Create dynamic plot."""
        # create figure
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.axis = plt.subplots(constrained_layout=True)
        self.artists = {}
        self.__displayed = False

        # configure figure canvas for ipympl
        if DynamicPlotter.interactive:
            self.fig.canvas.toolbar_position = "top"
            self.fig.canvas.header_visible = False

    def reset(self):
        """Reset dynamic plot."""
        # remove all artists
        self.artists = {}
        self.__displayed = False

    def display(self, out=None, clear=True, rescale=False):
        """Display or redraw plot depending on changes to set of artists."""
        if out is None:
            out = contextlib.nullcontext()
        
        if DynamicPlotter.interactive:
            if self.__displayed:
                # (optional) rescale plot automatically
                if rescale:
                    self.axis.relim()
                    self.axis.autoscale(tight=True)
                # redraw plot
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                # display plot
                with out:
                    display(self.fig.canvas, clear=clear)
                self.__displayed = True
        else:
            with out:
                display(self.fig, clear=clear)
            self.__displayed = True

    # PLOTTING METHODS

    def static(self, key, plt_f, visible=True):
        """Plot static artist on first call."""
        # display artist
        if key not in self.artists.keys():
            self.artists[key] = plt_f()
        # set visibility of artist
        self.set_visible(self.artists[key], visible)

    def dynamic(self, key, plt_f, update_f, visible=True):
        """Plot dynamic artist on first call and update on later calls."""
        # display or update artist
        if key not in self.artists.keys():
            self.artists[key] = plt_f()
        else:
            update_f(self.artists[key])
        # set visibility of artist
        self.set_visible(self.artists[key], visible)
    
    def dynamic_plot(self, key, *args, visible=True, **kwargs):
        """Plot dynamic Line2D artist on first call and update on later calls."""
        self.dynamic(
            key,
            lambda: self.axis.plot(*args, **kwargs),
            lambda lines: [line.set_data(*args[2*i:2*i+2]) for i, line in enumerate(lines)],
            visible=visible,
        )
    
    def dynamic_patch_collection(self, key, patches, visible=True, **kwargs):
        """Plot dynamic patch collection on first call and update on later calls."""
        self.dynamic(
            key,
            lambda: self.axis.add_collection(PatchCollection(patches, **kwargs)),
            lambda collection: collection.set_paths(patches),
            visible=visible,
        )

    # HELPER METHODS

    def set_visible(self, item, visible):
        """Change visibility of plotted artists."""
        if isinstance(item, plt.Artist):
            item.set_visible(visible)
        elif isinstance(item, list):
            for subitem in item:
                subitem.set_visible(visible)
        else:
            print("WARNING: not able to change visibility of {}".format(item))
