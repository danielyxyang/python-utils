__all__ = ["Plotter", "DynamicPlotter"]

import os
import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display

from .plot_tools import FilterTicksLocator


def _call_set_f(set_f, arg):
    """Call function either with first argument or keyword arguments."""
    if isinstance(arg, dict): set_f(**arg) # arg = keyword arguments
    else:                     set_f(arg)   # arg = first argument


def _warn_incorrect_layout_engine(func, fig):
    print("WARNING: {} does not support layout engine {}.".format(func.__name__, type(fig.get_layout_engine()).__name__))


def _set_figheight_auto(fig, prec=0.01, prec_mode="abs", max_iter=10, verbose=False):
    """Automatically sets figure height tightest possible while adhering to padding.
    
    This function retrieves the desired padding from the figure's layout engine
    and computes a new height based on the height of the figure's tight bounding
    box. Since the layout engines often themselve rely on the current figure
    height, the figure height is updated iteratively until it converges.
    """
    layout_engine = fig.get_layout_engine()
    if isinstance(layout_engine, mpl.layout_engine.TightLayoutEngine):
        # compute padding in inches for tight layout engine
        # https://github.com/matplotlib/matplotlib/blob/v3.7.1/lib/matplotlib/_tight_layout.py#L50
        font_size_inch = mpl.font_manager.FontProperties(size=mpl.rcParams["font.size"]).get_size_in_points() / 72
        h_pad_inch = layout_engine.get()["pad"] * font_size_inch
    elif isinstance(layout_engine, mpl.layout_engine.ConstrainedLayoutEngine):
        # compute padding in inches for constrained layout engine
        h_pad_inch = layout_engine.get()["h_pad"]
        # switch to compressed constrained layout engine to remove white space (faster convergence)
        layout_engine._compress = True
        layout_engine.execute(fig)
    else:
        _warn_incorrect_layout_engine(_set_figheight_auto, fig)
    
    # initialize figure height to some sufficiently large height
    fig.set_figheight(50)
    layout_engine.execute(fig)
    if verbose:
        print("{:5.2f}".format(fig.get_figheight()))
    # iteratively update figure height until layout engine converges
    for _ in range(max_iter):
        # compute new figure height
        new_height = fig.get_tightbbox().height + 2 * h_pad_inch
        if verbose:
            print("{:5.2f} {:6.3f} {:6.1%}".format(new_height, new_height - fig.get_figheight(), new_height / fig.get_figheight() - 1))
        # check early stopping
        if (
            (prec_mode == "abs" and np.abs(new_height - fig.get_figheight()) < prec)
            or (prec_mode == "rel" and np.abs(new_height / fig.get_figheight() - 1) < prec)
        ):
            if verbose:
                print("Early stopping with", fig.get_figheight())
            break
        # update figure height and recalculate layout
        fig.set_figheight(new_height)
        layout_engine.execute(fig)


def _get_padding(fig):
    """Get the figure's padding in inches.
    
    This function relies on the subplot parameters and therefore only works with
    the tight layout engine.
    """
    if isinstance(fig.get_layout_engine(), mpl.layout_engine.ConstrainedLayoutEngine):
        _warn_incorrect_layout_engine(_set_padding, fig)
    return np.array([
        fig.subplotpars.left * fig.get_figwidth(), # left
        fig.subplotpars.bottom * fig.get_figheight(), # bottom
        (1 - fig.subplotpars.right) * fig.get_figwidth(), # right
        (1 - fig.subplotpars.top) * fig.get_figheight(), # top
    ])


def _set_padding(fig, left, bottom, right, top):
    """Set the figure's padding in inches.
    
    This function relies on the subplot parameters and therefore only works with
    the tight layout engine.
    """
    if isinstance(fig.get_layout_engine(), mpl.layout_engine.ConstrainedLayoutEngine):
        _warn_incorrect_layout_engine(_set_padding, fig)
    fig.subplots_adjust(
        left=left / fig.get_figwidth(),
        bottom=bottom / fig.get_figheight(),
        right=1 - right / fig.get_figwidth(),
        top=1 - top / fig.get_figheight(),
    )


def _execute_tight_layout_auto(fig, prec=0.01, max_iter=10, verbose=False):
    """Iteratively executes tight layout engine until convergence.
    
    The tight layout engine seems to rely on the current layout of the figure.
    Hence, the layout engine is executed iteratively until the adjusted subplot
    parameters stabilize, which specify the figure's paddings. A related issue
    is described here [1].
    
    References:
        [1] https://github.com/matplotlib/matplotlib/issues/11809
    """
    layout_engine = fig.get_layout_engine()
    if not isinstance(layout_engine, mpl.layout_engine.TightLayoutEngine):
        _warn_incorrect_layout_engine(_execute_tight_layout_auto, fig)
        return

    padding = _get_padding(fig)
    if verbose:
        print("{}".format(padding))
    for _ in range(max_iter):
        # execute layout engine and compute new padding
        layout_engine.execute(fig)
        padding_new = _get_padding(fig)
        if verbose:
            print("{} {:6.3f}".format(padding_new, np.max(np.abs(padding - padding_new))))
        # check early stopping
        if (np.abs(padding - padding_new) < prec).all():
            if verbose:
                print("Early stopping with", padding_new)
            break
        # update previous padding
        padding = padding_new
    

class Plotter():
    r"""Extend and simplify the process of creating, showing and saving plots.
    
    The main features include:
        - Configure Matplotlib's style-relevant rcParams more simpler. 
        - Set properties of axes using a more powerful set-method.
        - Set figure size based on figure width and axes ratio.
        - Show interactive plots in notebooks.
        - Save plots with consistent size.
        - Save PDF/PGF/TikZ plots rendered with LaTeX.

    Examples:
        ```
        Plotter.setup(interactive=True, is_colab=False)
        Plotter.configure(
            basewidth=BASEWIDTH,
            latex=True,
            latex_preamble="\n".join([
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{lmodern}",      # for 8-bit Latin Modern font
                r"\usepackage[sc]{mathpazo}", # for Palatino font
                r"\usepackage{amsmath,amssymb}",
            ]),
            save_dir=os.path.join(".", "output"),
            save_format="pdf",
        )

        fig, axes = Plotter.create(ncols=2, set=dict(
            xlabel="x", ylabel="y",
            centeraxes=True,
        )
        Plotter.set(axes[0], title="My First Plot", xlim=(-3, 3), ylim=(2, 4))
        # CODE FOR PLOT 1
        Plotter.set(axes[1], title="My Second Plot")
        # CODE FOR PLOT 2
        Plotter.set(axes, legend=True)
        Plotter.finish((fig, "FILENAME"), figwidth=0.5, save=True)
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
        """Setup environment for Plotter.

        Args:
            interactive (bool, optional): Flag whether the plots should be
                interactive or not. This only works in an interactive IPython
                environment. Defaults to None.
            is_colab (bool, optional): Flag whether the Plotter is used in
                Google Colab or not. This is required for proper functioning of
                interactive plots in Colab. Defaults to None.
        """
        if interactive is not None: Plotter.interactive = interactive
        if is_colab is not None:    Plotter.is_colab = is_colab
        
        # enable interactive plots in IPython environment
        ip = get_ipython()
        if ip is not None:
            # setup matplotlib backend with IPython magic
            if Plotter.interactive:
                ip.run_line_magic("matplotlib", "ipympl")
            else:
                ip.run_line_magic("matplotlib", "inline")
                # display images without cropping
                # https://github.com/jupyter/notebook/issues/2640#issuecomment-579369065
                ip.run_line_magic("config", "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")

            # enable interactive plots on Colab
            # https://matplotlib.org/ipympl/installing.html#google-colab
            if Plotter.interactive and Plotter.is_colab:
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
        """Configure default behavior of Plotter.

        The default style of the plots can be configured with Matplotlib's
        rcParams and style sheets. The default rcParams can be found here [1]
        and a list of different built-in style sheets here [2].
        
        Args:
            basewidth (float, optional): The basewidth of the area in which the
                plots are used in inches. This can be the width of the output
                area or the textwidth of your LaTeX document, which can be
                obtained with "\the\textwidth" divided by 72.27 [3]. Defaults to
                None.
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
            [3] https://www.overleaf.com/learn/latex/Lengths_in_LaTeX
        """
        if basewidth is not None:   Plotter.basewidth = basewidth
        if save_dir is not None:    Plotter.save_dir = save_dir
        if save_format is not None: Plotter.save_format = save_format
        if save_always is not None: Plotter.save_always = save_always

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
                    for _, key in Plotter._FONTSIZE_KEYS.items()
                })
            elif isinstance(fontsize, dict):
                default = fontsize.get("default", None)
                plt.rcParams.update({
                    key: fontsize.get(key_short, default)
                    for key_short, key in Plotter._FONTSIZE_KEYS.items()
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
        """Configure default behavior of Plotter with context manager."""
        try:
            # save parameters of Plotter
            prev = (
                Plotter.basewidth,
                Plotter.save_dir,
                Plotter.save_format,
                Plotter.save_always,
            )
            # save and restore rcParams of Matplotlib
            with plt.rc_context():
                # change configuration
                Plotter.configure(**kwargs)
                yield
        finally:
            # restore parameters of Plotter
            (
                Plotter.basewidth,
                Plotter.save_dir,
                Plotter.save_format,
                Plotter.save_always,
            ) = prev
    
    # PLOTTING FUNCTIONS

    @staticmethod
    def create(layout=dict(layout="tight", pad=0.25), set={}, subplot={}, gridspec={}, **kwargs):
        """Create a new plot.

        The two main layout engines [1] are "tight" and "constrained". The main
        differences [2] are:
            - The tight layout engine adjusts the subplot parameters, such that
            all titles, labels and ticks fit into the given figure size. If
            multiple subplots exist, they are squeezed into a tight group
            centered in space. The parameters `pad`, `w_pad` and `h_pad` must be
            specified as a fraction of the font size.
            - The constrained layout engine adjusts the axes size using layout
            grids, such that all titles, labels and ticks fit into the given 
            figure size. If multiple subplots exist, they are distributed across
            space unless `compress` is set to True. The parameters `w_pad` and
            `h_pad` must be specified in inches (?).
        Note that some functionalities of Plotter only work with the tight
        layout engine.
        
        Args:
            layout (str or dict): The layout engine to be used for the figure.
                Additional layout engine parameters [1] can be specified by
                providing a dict with the layout engine name specified under the
                key "layout". Defaults to dict(layout="tight", pad=0.25).
            set (dict): Keyword arguments passed to `set`. Defaults to {}.
            subplot (dict): Keyword arguments passed to `add_subplot`. Defaults
                to {}.
            gridspec (dict): Keyword arguments passed to `GridSpec`. Defaults to
                {}.
            **kwargs: Keyword arguments passed to `subplots` function [3].
        
        Returns:
            Tuple of Figure and Axes instance.
        
        References:
            [1] https://matplotlib.org/stable/api/layout_engine_api.html
            [2] https://stackoverflow.com/a/72204558
            [3] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
        """
        if isinstance(layout, dict):
            layout_name = layout["layout"]
            layout_params = {k: v for k, v in layout.items() if k != "layout"}
        else:
            layout_name = layout
            layout_params = {}
        
        # create figure and axes
        fig, axes = plt.subplots(layout=layout_name, subplot_kw=subplot, gridspec_kw=gridspec, **kwargs)
        
        # set layout engine parameters
        fig.get_layout_engine().set(**layout_params)

        # set axes properties
        Plotter.set(axes, **set)

        # configure style for interactive plots
        if Plotter.interactive:
            fig.canvas.toolbar_position = "top"
            fig.canvas.header_visible = False
        
        return fig, axes

    @staticmethod
    def set(
        axes,
        title=None,
        xlabel=None, ylabel=None,
        xlim=None, ylim=None,
        xmargin=None, ymargin=None,
        xticks=None, yticks=None,
        centeraxes=False,
        legend=False,
        **kwargs,
    ):
        """Set multiple properties for axes at once.

        Args:
            axes (Axes or list of Axes): The matplotlib Axes on which the
                properties should be applied.
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
            xticks (list or dict, optional): List of xticks or dict with
                arguments for `set_xticks`. If the dict is of the form
                dict(locator=X, formatter=X), the major locator and formatter
                are set instead. Defaults to None.
            yticks (list or dict, optional): List of yticks or dict with
                arguments for `set_yticks`. If the dict is of the form
                dict(locator=X, formatter=X), the major locator and formatter
                are set instead. Defaults to None.
            centeraxes (bool, optional): Flag whether to center the axes.
                Defaults to False.
            legend (bool or dict, optional): Flag whether to show the legend
                with default options or dict with arguments for `legend` and
                additional "order" keyword for specifying the order of artists
                by position. Defaults to False.
            **kwargs: Keyword arguments passed to `set` [1].
        
        References:
            [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
        """
        for axis in np.asarray(axes).flatten():
            # set text
            if title is not None:  _call_set_f(axis.set_title, title)
            if xlabel is not None: _call_set_f(axis.set_xlabel, xlabel)
            if ylabel is not None: _call_set_f(axis.set_ylabel, ylabel)
            # set ticks (before limits)
            if xticks is not None:
                if isinstance(xticks, dict) and "locator" in xticks and "formatter" in xticks:
                    axis.xaxis.set_major_locator(xticks["locator"])
                    axis.xaxis.set_major_formatter(xticks["formatter"])
                else:
                    _call_set_f(axis.set_xticks, xticks)
            if yticks is not None:
                if isinstance(yticks, dict) and "locator" in yticks and "formatter" in yticks:
                    axis.yaxis.set_major_locator(yticks["locator"])
                    axis.yaxis.set_major_formatter(yticks["formatter"])
                else:
                    _call_set_f(axis.set_yticks, yticks)
            # set limits
            if xlim is not None:      _call_set_f(axis.set_xlim, xlim)
            elif xmargin is not None: _call_set_f(axis.set_xmargin, xmargin)
            if ylim is not None:      _call_set_f(axis.set_ylim, ylim)
            elif ymargin is not None: _call_set_f(axis.set_ymargin, ymargin)
            # center axes
            if centeraxes:
                # center left and bottom axes
                axis.spines["left"].set_position("zero")
                axis.spines["bottom"].set_position("zero")
                # hide upper and right axes
                axis.spines["right"].set_visible(False)
                axis.spines["top"].set_visible(False)
                # add arrow tips to left and bottom axes
                axis.plot(1, 0, ">", color="black", transform=axis.get_yaxis_transform(), clip_on=False)
                axis.plot(0, 1, "^", color="black", transform=axis.get_xaxis_transform(), clip_on=False)
                # show ticks only on left and bottom axes
                axis.xaxis.set_ticks_position("bottom")
                axis.yaxis.set_ticks_position("left")
                # hide ticks at origin
                axis.xaxis.set_major_locator(FilterTicksLocator(axis.xaxis.get_major_locator(), [0]))
                axis.yaxis.set_major_locator(FilterTicksLocator(axis.yaxis.get_major_locator(), [0]))
                # move xlabel and ylabel
                axis.xaxis.set_label_text(axis.get_xlabel(), ha="right", va="bottom")
                axis.xaxis.set_label_coords(1, 0.05, transform=axis.get_yaxis_transform())
                axis.yaxis.set_label_text(axis.get_ylabel(), ha="left", va="top", rotation="horizontal")
                axis.yaxis.set_label_coords(0.2, 1, transform=axis.get_xaxis_transform())
            # set legend
            if legend:
                if isinstance(legend, dict):
                    if "order" in legend:
                        # apply permutation to order of legend entries
                        order = legend.pop("order")
                        handles, labels = axis.get_legend_handles_labels()
                        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
                        legend["handles"] = handles
                        legend["labels"] = labels
                    axis.legend(**legend)
                else:
                    axis.legend()
            # set remaining properties
            axis.set(**kwargs)

    @staticmethod
    def finish(
        plots,
        # parameters for figure size
        figsize=None,
        figwidth=1, axratio=None,
        figsize_unit="base",
        consistent_size=False,
        # parameters for other properties
        set={},
        # parameters for displaying figures
        show=True, show_ncols=4,
        # parameters for saving figures
        save=False, save_format=None, save_kw={},
    ):
        """Finish plots by displaying and/or saving them.

        There are two ways to specify the figure size, prioritized in the
        following order:
            1) Provide the size of the figure as tuple (width, height) using the
            parameter figsize.
            2) Provide the width of the figure and ratio of the axes using the
            parameters figwidth and axratio. The height of the figure is then
            determined automatically.
        
        In order to use consistent_size the figure size must be provided
        according to 2) and all plots must use the tight layout engine. The
        reason is that consistent_size relies on the fig.subplotpars which are
        only adjusted by tight layout.

        Args:
            plots (Figure, tuple (Figure, str) or mixed list of Figures and
                tuples): The plot or list of plots to be displayed and/or saved.
                If provided as a tuple (figure, name), the figure is displayed
                and saved under the given name.
            figsize (tuple (float, float), optional): The tuple (width, height)
                with width and height given in figsize_unit. Defaults to None.
            figwidth (float, optional): The width of the figure given in
                figsize_unit. This parameter is ignored if figsize is given.
                Defaults to 1.
            axratio (float, optional): The height/width ratio of the axes, not
                of the figure. This parameter is ignored if figsize is given.
                Defaults to None (golden mean ratio).
            figsize_unit ("base", "inch" or "cm"): The unit in which the width
                and/or height of the figure are specified. The size can be
                relative to Plotter.basewidth ("base"). Defaults to "base".
            consistent_size (bool, optional): Flag whether to consistently size
                the Axes across different figures by unifying their side
                paddings. Defaults to False.
            set (dict): Keyword arguments passed to `set`. Defaults to {}.
            show (bool, optional): Flag whether to display the plots. Defaults
                to True.
            show_ncols (int, optional): Number of columns for displaying the
                list of plots. Defaults to 4.
            save (bool, optional): Flag whether to save the plots. Defaults to
                False.
            save_format (str, optional): The format in which the plots should be
                saved. Formats include "png", "pdf", "pgf" and "tikz". Saving as
                "tikz" requires the package `tikzplotlib`. Defaults to None.
            save_kw (dict): Keyword arguments passed to the corresponding save
                function. Defaults to {}.
        """
        # ensure plots is a list of tuples
        if not isinstance(plots, list):
            plots = [plots]
        if len(plots) == 0:
            return
        # ensure plots are named
        plots = [
            plot if plot is None or isinstance(plot, tuple) else (plot, "plot{}".format(plot.number))
            for plot in plots
        ]
        # define list of non-empty plots for simpler for-loops (aliasing plots)
        plots_filtered = [plot for plot in plots if plot is not None]

        # set figure size unit
        if figsize_unit == "base":
            figsize_unit = Plotter.basewidth
        elif figsize_unit == "inch":
            figsize_unit = 1
        elif figsize_unit == "cm":
            figsize_unit = 1 / 2.54
        else:
            print("WARNING: figsize_unit \"{}\" is unknown.".format(figsize_unit))

        # set figure size specification
        if figsize is not None:
            figsize_spec = dict(
                spec="width_height",
                width=figsize[0] * figsize_unit,
                height=figsize[1] * figsize_unit,
            )
        elif figwidth is not None:
            figsize_spec = dict(
                spec="width_ratio",
                width=figwidth * figsize_unit,
                ratio=axratio if axratio is not None else (np.sqrt(5.0) - 1.0) / 2.0, # golden mean
            )
        else:
            figsize_spec = dict(spec=None)
        
        # set figure size and axis properties
        for fig, _ in plots_filtered:
            # set figure size
            if figsize_spec["spec"] == "width_height":
                # set figure size based on width and height
                fig.set_size_inches(figsize_spec["width"], figsize_spec["height"])
            elif figsize_spec["spec"] == "width_ratio":
                # set figure size based on width and axes ratio
                for axis in fig.axes:
                    axis.set_box_aspect(figsize_spec["ratio"])
                fig.set_figwidth(figsize_spec["width"])
                _set_figheight_auto(fig)
            # set properties of axes
            Plotter.set(fig.axes, **set)
        
        # set consistent sizes (e.g. for side-by-side plots)
        if consistent_size:
            if figsize_spec["spec"] != "width_ratio":
                print("WARNING: consistent_size only works with specified figure width and axis ratio.")
            # execute tight layout to obtain proper subplot parameters
            for fig, _ in plots_filtered:
                if isinstance(fig.get_layout_engine(), mpl.layout_engine.TightLayoutEngine):
                    _execute_tight_layout_auto(fig)
                else:
                    print("WARNING: consistent_size only works with tight layout engine.")
            # compute max figure height
            height_max = np.max([fig.get_figheight() for fig, _ in plots_filtered])
            # compute max required padding for title, labels, ticks, etc.
            l_max, b_max, r_max, t_max = np.max([_get_padding(fig) for fig, _ in plots_filtered], axis=0)
            # set max figure height and required padding for all plots
            for fig, _ in plots_filtered:
                # set max figure height
                fig.set_figheight(height_max)
                # recalculate layout and then turn off tight layout engine (otherwise paddings are changed again)
                fig.get_layout_engine().execute(fig)
                fig.set_layout_engine("none")
                # set max required padding
                _set_padding(fig, left=l_max, bottom=b_max, right=r_max, top=t_max)

        # show figures
        if show or Plotter.save_always:
            # create grid of figures
            if figsize_spec["spec"] is not None:
                grid_width = figsize_spec["width"]
            else:
                grid_width = np.max([fig.get_figwidth() for fig, _ in plots_filtered])
            grid = widgets.GridspecLayout(
                n_rows=int(np.ceil(len(plots)/show_ncols)),
                n_columns=show_ncols,
                width="{}in".format((grid_width+0.5)*show_ncols),
            )
            for i, (fig, name) in enumerate(plots):
                if fig is None:
                    continue
                out = widgets.Output(layout=dict(overflow="auto"))
                with out:
                    if Plotter.interactive and not Plotter.save_always:
                        display(fig.canvas)
                    else:
                        display(fig)
                    display(widgets.Label(
                        "{:3}: {}".format(fig.number, name),
                        layout=dict(overflow="auto"),
                        style=dict(font_family="monospace", font_size="10pt"),
                    ))
                grid[i // show_ncols, i % show_ncols] = out
            # display grid of figures
            Plotter.display_html_hack()
            display(grid)

        # save figures
        if save or Plotter.save_always:
            if save_format is None or Plotter.save_always:
                save_format = Plotter.save_format
            if save_format in ["png", "tikz", "pgf", "pdf"]:
                for fig, name in plots_filtered:
                    # setup output path
                    filepath = os.path.join(Plotter.save_dir, save_format, name)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    # compute width and height annotations
                    width_in, height_in = fig.get_size_inches()
                    if figsize_unit == Plotter.basewidth:
                        width_latex = r"{:.2f}\textwidth".format(width_in / Plotter.basewidth)
                        height_latex = r"{:.2f}\textwidth".format(height_in / Plotter.basewidth)
                    else:
                        width_latex = "{:.2f}in".format(width_in)
                        height_latex = "{:.2f}in".format(height_in)
                    
                    # save figure to file
                    if save_format == "png":
                        filepath += ".png"
                        fig.savefig(filepath, **save_kw)
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print(r"\includegraphics[width=%s]{%s}" % (width_latex, name))
                    elif save_format == "pdf":
                        filepath += ".pdf"
                        fig.savefig(filepath, backend="pgf", **save_kw)
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print(r"\includegraphics[width=%s]{%s.pdf}" % (width_latex, name))
                    elif save_format == "pgf":
                        filepath += ".pgf"
                        fig.savefig(filepath, backend="pgf", **save_kw)
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print(r"\resizebox{%s}{!}{\input{%s.pgf}}" % (width_latex, name))
                    elif save_format == "tikz":
                        import tikzplotlib as tikz
                        filepath += ".tex"
                        tikz.save(filepath, axis_width=r"\tikzwidth", axis_height=r"\tikzheight", wrap=False, **save_kw)
                        print("Plot saved to \"{}\". Include in LaTeX with:".format(filepath))
                        print("\n".join([
                            r"\begin{tikzpicture}",
                            r"    \def\tikzwidth{%s}",
                            r"    \def\tikzheight{%s}",
                            r"    \input{%s}",
                            r"\end{tikzpicture}",
                        ]) % (width_latex, height_latex, name))
            else:
                print("WARNING: save_format \"{}\" unknown.".format(save_format))

    # UTILITY FUNCTIONS

    @staticmethod
    def display_html_hack():
        """Display CSS hack to show toolbar of interactive plots in Colab."""
        if Plotter.interactive and Plotter.is_colab:
            html_hack = widgets.HTML("<style> .jupyter-matplotlib-figure { position: relative; } </style>")
            display(html_hack)

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
