import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import mplcursors
from math import floor, ceil
import numpy as np
import warnings
from matplotlib.tri import Triangulation
import numbers

# Bandaide for mplcursors warning we don't need
warnings.filterwarnings("ignore", message="Pick support for PolyCollection is missing.")

default_hue_palette = sns.color_palette("viridis", as_cmap=True)


def validate_color(value):
    """
    Validate if the provided value is a valid color.

    Args:
        value (str): The color value to validate.

    Returns:
        bool: True if the value is a valid color, False otherwise.
    """
    try:
        colors.to_rgb(value)
        return True
    except ValueError:
        return False


def validate_marker(value):
    """
    Validate if the provided value is a valid marker.

    Args:
        value (str): The marker value to validate.

    Returns:
        bool: True if the value is a valid marker, False otherwise.
    """
    valid_markers = ['o', 's', 'D', 'd', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'x', 'X', '+', '|', '_', ".", None]
    return value in valid_markers


def validate_linestyle(value):
    """
    Validate if the provided value is a valid linestyle.

    Args:
        value (str): The linestyle value to validate.

    Returns:
        bool: True if the value is a valid linestyle, False otherwise.
    """
    valid_linestyles = ['-', '--', '-.', ':', 'None', ' ', '', None, False]
    return value in valid_linestyles


class Dataset:
    """
    A class to represent a dataset and manage its plotting attributes.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the data.
        title (str): The title of the dataset.
        index (int): The index of the dataset for differentiation.
        query (str): The query to filter the DataFrame.
        select (bool): The selection status of the dataset.
        color (str): The color used for plotting.
        marker (str): The marker style used for plotting.
        edge_color (str): The edge color of the markers.
        linestyle (str): The line style used for plotting.
        markersize (int): The size of the markers.
        alpha (float): The transparency level of the plot elements.
        hue (str): The column used for hue differentiation.
        hue_palette (str): The palette used for hue differentiation.
        hue_order (list): The order of hue levels.
        reg_order (int): The order of the regression fit.
        style (str): The style of the plot.
        linewidth (float): The width of the lines.
        edgewidth (float): The width of the marker edges.
        set_type (str): The type of the dataset (normal, delta, etc.)
        order (col): Column used to order/sort 
        delta_sets (tuple): If it's a delta set, the tuple of the two datasets with the first being the base.
    
    Methods:
        sel_query(query): Set the query for filtering the DataFrame.
        update_format_dict(format_options): Update multiple formatting options.
        get_format_dict(): Get the current formatting options as a dictionary.
        set_format_option(key, value): Set a specific formatting option.
        get_title(): Get the title of the dataset.
        delta_with(dataset, args): Create a delta dataset with another dataset.
    """

    def __init__(self, df, index=0, title=None, display_parms=None):
        """
        Initialize the Dataset object.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            index (int): The index of the dataset for differentiation. Default is 0.
            title (str): The title of the dataset. Default is None.
        """
        self._df_full = df
        self._df_filtered = df
        self.query = None
        self._select = True
        self.title = self.set_title = title if title else df["TITLE"].iloc[0] if "TITLE" in df.columns else "Untitled"
        self.index = index
        self.title_format = f"{self.title} {index}"
        self._color = cm.tab10(index % 10)
        self._marker = marker_map(index)
        self._edge_color = "black"
        self._linestyle = None
        self.markersize = self.sym_size = 12
        self.alpha = 1
        self.hue = self.color_on = None
        self.hue_palette = default_hue_palette
        self.hue_order = None
        self.reg_order = self.fit = None
        self.style = None
        self.linewidth = 2   # Default line width
        self.edgewidth = 1   # Default edge width for markers
        self.set_type = 1   # 1 = normal, 2 = delta, 3 = delta with fit
        self.delta_sets = None #  tuple of datasets for delta set
        self._display_parms = display_parms if display_parms else []
        self._plot_type = 'scatter'
        self._order = None  # Initialize the _order attribute

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):

        if (value in self._df_full.columns) or (value==None):
            self._order = value
        else:
            raise ValueError(f"Invalid order column: {value}. Column does not exist in DataFrame.")

    @property
    def df(self):
        return self._df_filtered

    @df.setter
    def df(self, value):
        self._df_full = value
        self._apply_query()

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, value):
        self._query = value
        self._apply_query()

    def _apply_query(self):
        """
        Apply the current query to _df_full and update _df_filtered.
        """
        if not self._query:
            self._df_filtered = self._df_full
        else:
            try:
                result_df = self._df_full.query(self._query)
                if not result_df.empty:
                    self._df_filtered = result_df
                else:
                    print(f"No data in set {self.index} after query: {self._query}. Turning Set Off...")
                    self.select = False
                    self._df_filtered = self._df_full
            except Exception as e:
                raise ValueError(f"Query error: {e}")

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if validate_color(value):
            self._color = value
        else:
            raise ValueError(f"Invalid color value: {value}")
        
    @property
    def select(self):
        return self._select

    @select.setter
    def select(self, value):
        if value in [True, 'True', 'true', 1, 't', 'T', 'on', 'On', 'ON']:
            self._select = True
        elif value in [False, 'False', 'false', 0, 'f', 'F', 'off', 'Off', 'OFF']:
            self._select = False
        else:
            raise ValueError(f"Invalid value for on: {value}")

    @property
    def edge_color(self):
        """
        Color the edge of the markers
        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value):
        """
        Set the edge color of the markers.
        """
        if validate_color(value):
            self._edge_color = value
        else:
            raise ValueError(f"Invalid color value: {value}")

    @property
    def plot_type(self):
        """
        Get the plot_type style used for plotting.

        Returns:
            str: The plot_type style.
        """
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value):
        """
        Set the plot_type style used for plotting.
        """
        valid_plot_types = ['scatter', 'contour']
        if value in valid_plot_types:
            self._plot_type = value
        else:
            raise ValueError(f"Invalid plot_type value: {value}")

    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, value):
        if validate_marker(value):
            self._marker = value
        else:
            raise ValueError(f"Invalid marker value: {value}")

    @property
    def linestyle(self):
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        if validate_linestyle(value):
            self._linestyle = value
        else:
            raise ValueError(f"Invalid linestyle value: {value}")

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._linewidth = value
        else:
            raise ValueError(f"Invalid linewidth value: {value}")

    @property
    def edgewidth(self):
        return self._edgewidth

    @edgewidth.setter
    def edgewidth(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._edgewidth = value
        else:
            raise ValueError(f"Invalid edgewidth value: {value}")

    @property
    def display_parms(self):
        return self._display_parms

    @display_parms.setter
    def display_parms(self, value):
        if isinstance(value, list):
            self._display_parms = value
        else:
            raise ValueError(f"display_parms must be a list, got {type(value)}")

    def sel_query(self, query):
        self.query = query

    def update_format_dict(self, format_options):
        for key, value in format_options.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid format key: {key}")

    def get_format_dict(self):
        return {
            'title': self.title,
            'color': self.color,
            'marker': self.marker,
            'edge_color': self.edge_color,
            'linestyle': self.linestyle,
            'markersize': self.markersize,
            'alpha': self.alpha,
            'hue': self.hue,
            'hue_palette': self.hue_palette,
            'hue_order': self.hue_order,
            'reg_order': self.reg_order,
            'index': self.index,
            'style': self.style,
            'display_parms': self.display_parms,
            'plot_type': self.plot_type,
            'linewidth': self.linewidth,
            'edgewidth': self.edgewidth
        }

    def set_format_option(self, key, value):
        """
        Set a specific formatting option.

        Args:
            key (str): The format option key.
            value: The format option value.

        Raises:
            ValueError: If the format key is invalid.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ValueError(f"Invalid format key: {key}")

    def get_title(self):
        """
        Get the title of the dataset.

        Returns:
            str: The title of the dataset.
        """
        return self.title

def table_read(df, x_col, y_col, x_in):
    """
    Perform interpolation on a table.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x values.
        y_col (str): The column name for the y values.
        x_in (array-like): The input x values for interpolation.

    Returns:
        np.ndarray: The interpolated y values.
    """
    df_sorted = df.sort_values(by=x_col)
    f = interp1d(df_sorted[x_col], df_sorted[y_col], fill_value="extrapolate")
    y_interp = f(x_in)
    return y_interp


def uniplot(list_of_datasets, x, y, z=None, plot_type=None, color=None, hue=None, marker=None, 
            markersize=12, marker_edge_color="black", hue_palette=default_hue_palette, 
            hue_order=None, line=False, suppress_msg=False, 
            return_axes=False, axes=None, suptitle=None, dark_mode=False, interactive=True,
            display_parms=None, grid=True, legend='above', legend_ncols=1, figsize=None,
            x_lim=None, y_lim=None):
    """
    Create a unified plot for a list of datasets.

    Args:
        list_of_datasets (list): List of Dataset objects to plot.
        x (str): The x-axis column name.
        y (str): The y-axis column name.
        color (str, optional): The color for the plot. Default is None.
        hue (str, optional): The column name for hue differentiation. Default is None.
        marker (str, optional): The marker style for the plot. Default is None.
        markersize (int, optional): The size of the markers. Default is 12.
        marker_edge_color (str, optional): The edge color of the markers. Default is "black".
        hue_palette (str, optional): The palette for hue differentiation. Default is default_hue_palette.
        hue_order (list, optional): The order of hue levels. Default is None.
        line (bool, optional): If True, plot as a line plot. Default is False.
        suppress_msg (bool, optional): If True, suppress messages. Default is False.
        return_axes (bool, optional): If True, return the axes. Default is False.
        axes (matplotlib.axes.Axes, optional): Axes to plot on. Default is None.
        suptitle (str, optional): The overall title for the plot. Default is None.
        dark_mode (bool, optional): If True, enable dark mode. Default is False.
        interactive (bool, optional): If True, enable interactive mode with mplcursors. Default is True.
        display_parms (list, optional): List of parameters to display in the annotation. Default is None.
        grid (bool, optional): If True, display gridlines. Default is True.
        legend (str, optional): Position of the legend. Default is 'above'.
        legend_ncols (int, optional): Number of columns in the legend. Default is 1.
        figsize (tuple, optional): Size of the figure (width, height). Default is (10, 8).

    Returns:
        matplotlib.axes.Axes: The plot axes if return_axes is True.
    """

    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=100)
    else:
        fig = axes.figure

    if figsize is None:
        figsize = (10, 8)

    if dark_mode:
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#2E2E2E')
        axes.set_facecolor('#2E2E2E')
        axes.spines['bottom'].set_color('white')
        axes.spines['top'].set_color('white')
        axes.spines['right'].set_color('white')
        axes.spines['left'].set_color('white')
        axes.xaxis.label.set_color('white')
        axes.yaxis.label.set_color('white')
        axes.title.set_color('white')
        axes.tick_params(axis='x', colors='white')
        axes.tick_params(axis='y', colors='white')
        legend_text_color = 'white'
        if grid:
            axes.grid(color='white', which='major', linestyle='-', alpha=0.1, linewidth=0.5, zorder=0)
    else:
        plt.style.use('default')
        fig.patch.set_facecolor('white')
        axes.set_facecolor('white')
        axes.spines['bottom'].set_color('black')
        axes.spines['top'].set_color('black')
        axes.spines['right'].set_color('black')
        axes.spines['left'].set_color('black')
        axes.xaxis.label.set_color('black')
        axes.yaxis.label.set_color('black')
        axes.title.set_color('black')
        axes.tick_params(axis='x', colors='black')
        axes.tick_params(axis='y', colors='black')
        legend_text_color = 'black'
        if grid:
            axes.grid(color='black', which='major', linestyle='-', alpha=0.1, linewidth=0.5, zorder=0)

    if suptitle:
        fig.suptitle(suptitle, fontsize='xx-large')
    else:
        fig.suptitle(f"{x} vs {y}", fontsize='xx-large')

    notitle_count = 0
    

    for dataset in list_of_datasets:
        if dataset.select:
            df = dataset.df

            if dataset.order == 'index':
                df = df.sort_index()
                sort = True
            elif dataset.order:
                sort_order = dataset.order
                df = df.sort_values(by=sort_order)
                sort = False #turn off autosort in lineplot
            else:
                df = df.sort_index()
                sort = True

            if "TITLE" not in df.columns:
                df["TITLE"] = f"Default Title"
                notitle_count += 1

            if x not in df.columns:
                if x.upper() in df.columns:
                    x = x.upper()
                else:
                    print(f"Dataframe does not contain {x}. Skipping...")
                    continue
            if y not in df.columns:
                if y.upper() in df.columns:
                    y = y.upper()
                else:
                    print(f"Dataframe does not contain {y}. Skipping...")
                    continue

            title_dict = dataset.get_format_dict()
            title = title_dict.get('title', dataset.get_title())

            hue = title_dict.get('hue', hue)
            palette = title_dict.get('hue_palette', hue_palette)
            color = title_dict.get('color', color)
            marker = title_dict.get('marker', marker)
            edge_color = title_dict.get('edge_color', marker_edge_color)
            markersize = title_dict.get('markersize', markersize)
            hue_order = title_dict.get('hue_order')
            linestyle = title_dict.get('linestyle', None)
            linewidth = title_dict.get('linewidth', 2)
            edgewidth = title_dict.get('edgewidth', 1)
            alpha = title_dict.get('alpha', 1)
            style = title_dict.get('style')
            reg_order = title_dict.get('reg_order')
            index = title_dict.get('index', 0)
            plot_type = title_dict.get('plot_type', 0)

            if plot_type == 'scatter':
                if not linestyle:
                    if hue:
                        scatter = sns.scatterplot(data=df, x=x, y=y, hue=hue, marker=marker, ax=axes,
                                            label=f"{index}: {title} colored on {hue}", palette=palette,
                                            legend=False, edgecolor=edge_color, linewidth=edgewidth, zorder=index+1)
                        norm = plt.Normalize(df[hue].min(), df[hue].max())
                        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                        sm.set_array([])
                        fig.colorbar(sm, ax=axes)
                        axes.legend(prop={'size': 12})    
                    else:
                        sns.scatterplot(data=df, x=x, y=y, ax=axes, color=color, marker=marker, 
                                        alpha=alpha, style=style, label=f"{index}: {title}",
                                        edgecolor=edge_color, linewidth=edgewidth, zorder=index+1)
                    axes.collections[-1].set_sizes([markersize**2])
                    axes.legend(prop={'size': 12})
                else:
                    if hue:
                        print("Unichart doesn't currently support lineplots with hue")
                        sns.lineplot(data=df, x=x, y=y, ax=axes, color=color, 
                                     linestyle=linestyle, marker=None, alpha=alpha, style=style, 
                                     sort=sort, linewidth=linewidth)
                        sns.scatterplot(data=df, x=x, y=y, ax=axes, 
                                        color="black", linestyle=linestyle, marker=marker, alpha=alpha, 
                                        style=style, label=f"{index}: {title} colored on {hue}",
                                        hue=hue, legend=False, size=markersize, palette=palette, zorder=index+1,
                                        linewidth=edgewidth, edgecolor=edge_color)
                        axes.collections[-1].set_sizes([markersize**2])
                        axes.legend(prop={'size': 12})
                    else:
                        if isinstance(reg_order, numbers.Number) and reg_order > 0:
                            scatter_kws = {'s': markersize, 'edgecolor': marker_edge_color,  'alpha': alpha, 'linewidth': edgewidth}
                            line_kws = {'linewidth': linewidth, 'alpha': alpha, 'linestyle' : linestyle}
                            sns.regplot(x=x, y=y, ax=axes, scatter_kws=scatter_kws, line_kws=line_kws,
                                        color=color, marker=marker, label=f"{index}: {title} Fit LS {reg_order}", 
                                        order=reg_order, data=df) 
                        else:
                            sns.lineplot(data=df, x=x, y=y, ax=axes, color=color, linestyle=linestyle, markersize=markersize, 
                                        marker=marker, alpha=alpha, style=style, label=f"{index}: {title}", zorder=index+1, 
                                        sort=sort, markeredgecolor=edge_color, linewidth=linewidth)

                lines = axes.get_lines()
                for line in lines:
                    if line.get_label() == title:
                        line.set_markersize(markersize)
                        line.set_markeredgecolor(marker_edge_color)
            elif plot_type == 'contour':
                    X = df[x]
                    Y = df[y]
                    Z = df[z] if z else df[hue]

                    triang = Triangulation(X, Y)
                    contour = axes.tricontourf(triang, Z, cmap=hue_palette, linewidths=linestyle, alpha=alpha)
                    for i, coll in enumerate(contour.collections):
                        coll.set_label(f"{dataset.index} : {title} _contour_{i}")

                    if hue:
                        norm = plt.Normalize(df[hue].min(), df[hue].max())
                        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                        sm.set_array([])
                        fig.colorbar(sm, ax=axes)

        axes.set_xlabel(x, fontsize='x-large')
        axes.set_ylabel(y, fontsize='x-large')
        
        if legend=='above':
            axes.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                        loc='upper left',
                        ncols=legend_ncols)
        elif legend=='default':
            # axes.legend(ncols=legend_ncols)
            pass
        elif legend=='off':
            try:
                legend = axes = axes.get_legend()
                if legend is not None:
                    legend.remove()
            except Exception as e:
                print(f"Error: {e} while remove legend")
        else:
            pass


        if interactive:

            cursor = mplcursors.cursor(axes)

            @cursor.connect("add")
            def on_add(sel):
                selected_title = sel.artist.get_label()
                try:
                    set_number = int(selected_title.split(":")[0])
                except ValueError:
                    # Handle unexpected label formats
                    print(f"Unexpected label format: {selected_title}")
                    return
                selected_dataset = list_of_datasets[set_number]
                selected_df = selected_dataset.df

                annotation_text = f'Point: ({sel.target[0]:.2f}, {sel.target[1]:.2f})\nDataset {selected_dataset.index}: {selected_dataset.title}'
                effective_display_parms = display_parms if display_parms else dataset.display_parms

                if effective_display_parms:
                    header = '\n{:<25} {:<5}'.format('Parameter', 'Value')
                    annotation_text += header
                    annotation_text += '\n' + '-'*35

                    def add_parameter(parm, value, interp=False):
                        value_str = f'{value:.2f}' if isinstance(value, numbers.Number) else str(value)
                        interp_str = ' (interp)' if interp else ''
                        
                        # Adjust the format specifiers for left and right alignment
                        # Assuming 15 characters for parameters and 10 for values, adjust as necessary
                        formatted_line = f'{parm:<20} {value_str:>10}{interp_str}'
                        
                        nonlocal annotation_text
                        annotation_text += '\n' + formatted_line

                    if isinstance(sel.index, numbers.Integral):  # if Integer
                        for parm in effective_display_parms:
                            if parm in selected_df.columns:
                                value = selected_df[parm].iloc[sel.index]
                                add_parameter(parm, value)
                    elif isinstance(sel.index, numbers.Number):
                        try:
                            float_index = float(sel.index)
                            low_index = floor(float_index)
                            high_index = ceil(float_index)
                            for parm in effective_display_parms:
                                if parm in selected_df.columns:
                                    low_value = selected_df[parm].iloc[low_index]
                                    high_value = selected_df[parm].iloc[high_index]
                                    value = low_value + (float_index - low_index) * (high_value - low_value)
                                    add_parameter(parm, value, interp=True)
                        except Exception as e:
                            print(f"Error: {e}")
                            return
                    else:
                        print(f"Invalid index: {sel.index}, Type: {type(sel.index)}")
                        return
                    
                sel.annotation.set(text=annotation_text, color='black')
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        if x_lim:
            axes.set_xlim(x_lim)
        if y_lim:
            axes.set_ylim(y_lim)

        if return_axes:
            return axes

def marker_map(value):
    markers = ['o', 's', 'D',  'v', '^', '<', '>', 'd', 'H', 'p', '*']
    return markers[value % len(markers)]


if __name__ == "__main__":
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "TITLE": ["DF1"]*3})
    df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12], "TITLE": ["DF2"]*3})
    df3 = pd.DataFrame({"A": [7, 8, 9], "B": [15, 20, 30], "TITLE": ["DF3"]*3})

    setlist = [Dataset(df, index=i) for (i, df) in enumerate([df1, df2, df3])]

    # Valid color and marker
    try:
        setlist[0].color = "red"
        setlist[0].marker = "s"
        print(f"Color: {setlist[0].color}, Marker: {setlist[0].marker}")
    except ValueError as e:
        print(e)

    # Invalid color
    try:
        setlist[0].color = "invalid_color"
    except ValueError as e:
        print(e)

    # Invalid marker
    try:
        setlist[0].marker = "blue"
    except ValueError as e:
        print(e)

    fig, ax = plt.subplots()
    uniplot(setlist, x="A", y="B", axes=ax)
    plt.show()
