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

#Bandaide for mplcursors warning we don't need
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
    valid_markers = ['o', 's', 'D', 'd', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'x', 'X', '+', '|', '_']
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
        set_type (str): The type of the dataset (normal, delta, etc.)
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
        self.set_type = 1   # 1 = normal, 2 = delta, 3 = delta with fit
        self.delta_sets = None # tuple of datasets for delta set
        self._display_parms = display_parms if display_parms else []
        self._plot_type = 'scatter'

    @property
    def df(self):
        """
        Get the filtered DataFrame based on the query.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if not self.query:
            return self._df_full
        try:
            result_df = self._df_full.query(self.query)
            if not result_df.empty:
                return result_df
            else:
                print(f"No data in set {self.index} after query: {self.query}. Turning Set Off...")
                self.select = False
        except Exception as e:
            raise ValueError(f"Query error: {e}")

    @df.setter
    def df(self, value):
        """
        Set the DataFrame.

        Args:
            value (pd.DataFrame): The new DataFrame.
        """
        self._df_full = value

    @property
    def color(self):
        """
        Get the color used for plotting.

        Returns:
            str: The color value.
        """
        return self._color

    @color.setter
    def color(self, value):
        """
        Set the color used for plotting.

        Args:
            value (str): The new color value.

        Raises:
            ValueError: If the color value is invalid.
        """
        if validate_color(value):
            self._color = value
        else:
            raise ValueError(f"Invalid color value: {value}")
        
    @property
    def select(self):
        """
        Get the selection status of the dataset.

        Returns:
            bool: The selection status.
        """
        return self._select

    @select.setter
    def select(self, value):
        """
        Set the selection status of the dataset.

        Args:
            value (bool): The new selection status.

        Raises:
            ValueError: If the selection value is invalid.
        """
        if value in [True, 'True', 'true', 1, 't', 'T', 'on', 'On', 'ON']:
            self._select = True
        elif value in [False, 'False', 'false', 0, 'f', 'F', 'off', 'Off', 'OFF']:
            self._select = False
        else:
            raise ValueError(f"Invalid value for on: {value}")

    @property
    def edge_color(self):
        """
        Get the edge color of the markers.

        Returns:
            str: The edge color value.
        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value):
        """
        Set the edge color of the markers.

        Args:
            value (str): The new edge color value.

        Raises:
            ValueError: If the edge color value is invalid.
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

        Args:
            value (str): The new plot_type style.

        Raises:
            ValueError: If the marker style value is invalid.
        """
        valid_plot_types = ['scatter', 'contour']
        if value in valid_plot_types:
            self._plot_type = value
        else:
            raise ValueError(f"Invalid plot_type value: {value}")


    @property
    def marker(self):
        """
        Get the marker style used for plotting.

        Returns:
            str: The marker style.
        """
        return self._marker

    @marker.setter
    def marker(self, value):
        """
        Set the marker style used for plotting.

        Args:
            value (str): The new marker style.

        Raises:
            ValueError: If the marker style value is invalid.
        """
        if validate_marker(value):
            self._marker = value
        else:
            raise ValueError(f"Invalid marker value: {value}")

    @property
    def linestyle(self):
        """
        Get the linestyle style used for plotting.

        Returns:
            str: The linestyle style.
        """
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        """
        Set the linestyle style used for plotting.

        Args:
            value (str): The new linestyle style.

        Raises:
            ValueError: If the linestyle style value is invalid.
        """
        if validate_linestyle(value):
            self._linestyle = value
        else:
            raise ValueError(f"Invalid linestyle value: {value}")
    
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
        """
        Set the query for filtering the DataFrame.

        Args:
            query (str): The query string.
        """
        self.query = query

    def update_format_dict(self, format_options):
        """
        Update multiple formatting options.

        Args:
            format_options (dict): A dictionary of format options and their values.

        Raises:
            ValueError: If a format key is invalid.
        """
        for key, value in format_options.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid format key: {key}")

    def get_format_dict(self):
        """
        Get the current formatting options as a dictionary.

        Returns:
            dict: The current formatting options.
        """
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
            'plot_type': self.plot_type
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
    
    # def delta_with(self, dataset, delta_parms, delta_type='index', delta_on=None):

    #     df1 = self._df_full
        
    #     #interpret dataset
    #     if isinstance(dataset, Dataset):
    #         df2 = dataset._df_full
    #     elif isinstance(dataset, pd.DataFrame):
    #         df2 = dataset
    #     else:
    #         print("Invalid dataset")
    #         return
        
    #     #interpret delta_type
    #     if delta_type == 'index':



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
            hue_order=None, line=False, ignore_list=[], suppress_msg=False, 
            return_axes=False, axes=None, suptitle=None, dark_mode=False, interactive=True,
            display_parms=None, grid=True):

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
        ignore_list (list, optional): List of titles to ignore. Default is [].
        suppress_msg (bool, optional): If True, suppress messages. Default is False.
        return_axes (bool, optional): If True, return the axes. Default is False.
        axes (matplotlib.axes.Axes, optional): Axes to plot on. Default is None.

    Returns:
        matplotlib.axes.Axes: The plot axes if return_axes is True.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), dpi=100)
    else:
        fig = axes.figure

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
        df = dataset.df
        if "TITLE" not in df.columns:
            df["TITLE"] = f"Default Title"
            notitle_count += 1

    for dataset in list_of_datasets:
        if dataset.select:
            df = dataset.df
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

            if title in ignore_list:
                continue

            hue = title_dict.get('hue', hue)
            palette = title_dict.get('hue_palette', hue_palette)
            color = title_dict.get('color', color)
            marker = title_dict.get('marker', marker)
            edge_color = title_dict.get('edge_color', marker_edge_color)
            markersize = title_dict.get('markersize', markersize)
            hue_order = title_dict.get('hue_order')
            linestyle = title_dict.get('linestyle', None)
            alpha = title_dict.get('alpha', 1)
            style = title_dict.get('style')
            reg_order = title_dict.get('reg_order')
            index = title_dict.get('index', 0)
            plot_type = title_dict.get('plot_type', 0)

            if plot_type == 'scatter':
                if not linestyle:
                    if hue:
                        scatter = sns.scatterplot(data=df, x=x, y=y, hue=hue, marker=marker, ax=axes,
                                            label=f"{index} : {title} colored on {hue}", palette=palette,
                                            legend=False, edgecolor=edge_color, linewidth=2, zorder=index+1)
                        scatter.collections[-1].set_sizes([markersize**2])
                        norm = plt.Normalize(df[hue].min(), df[hue].max())
                        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                        sm.set_array([])
                        fig.colorbar(sm, ax=axes)
                        axes.legend(prop={'size': 12})    
                    else:
                        sns.scatterplot(data=df, x=x, y=y, ax=axes, color=color, marker=marker, 
                                        alpha=alpha, style=style, label=f"{index} : {title}",
                                        edgecolor=edge_color, linewidth=2, zorder=index+1)
                        axes.collections[-1].set_sizes([markersize**2])
                        axes.legend(prop={'size': 12})
                else:
                    if hue:
                        print("Unichart doesn't currently support lineplots with hue")
                        sns.scatterplot(data=df, x=x, y=y, ax=axes, color="black", linestyle=linestyle, 
                                        marker=marker, alpha=alpha, style=style, label=f"{index} : {title} colored on {hue}",
                                        hue=hue, legend=False, size=markersize, palette=palette, zorder=index+1)
                    else:
                        if isinstance(reg_order, (int, float)) and reg_order > 0:
                            scatter_kws = {'s': markersize**2, 'edgecolor': marker_edge_color,  'alpha': alpha}
                            line_kws = {'linewidth': 2, 'alpha': alpha, 'linestyle' : linestyle}
                            sns.regplot(x=x, y=y, ax=axes, scatter_kws=scatter_kws, line_kws=line_kws,
                                        color=color, marker=marker, label=f"{index} : {title} Fit LS {reg_order}", 
                                        order=reg_order, data=df.sort_values(by=x), zorder=index+1) 
                        else:
                            sns.lineplot(data=df, x=x, y=y, ax=axes, color=color, linestyle=linestyle, markersize=markersize, 
                                        marker=marker, alpha=alpha, style=style, label=f"{index} : {title} Fit ST", zorder=index+1)

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

                    if hue:
                        norm = plt.Normalize(df[hue].min(), df[hue].max())
                        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                        sm.set_array([])
                        fig.colorbar(sm, ax=axes)

        axes.set_xlabel(x, fontsize='x-large')
        axes.set_ylabel(y, fontsize='x-large')
        
        if interactive:

            cursor = mplcursors.cursor(axes)

            @cursor.connect("add")
            def on_add(sel):
                selected_title = sel.artist.get_label()
                set_number = int(selected_title.split()[0])
                selected_dataset = list_of_datasets[set_number]
                selected_df = selected_dataset.df

                annotation_text = f'Point: ({sel.target[0]:.2f}, {sel.target[1]:.2f})\nDataset: {selected_dataset.title}'
                effective_display_parms = display_parms if display_parms else dataset.display_parms

                if effective_display_parms:
                    header = '\n{:<15} {:<10}'.format('Parameter', 'Value')
                    annotation_text += header
                    annotation_text += '\n' + '-'*26  # Add a separator line

                    def add_parameter(parm, value, interp=False):
                        value_str = f'{value:.2f}' if isinstance(value, (int, float)) else f'{value}'
                        interp_str = ' (interp)' if interp else ''
                        nonlocal annotation_text
                        annotation_text += '\n{:<15} {:<10}{}'.format(parm, value_str, interp_str)

                    if isinstance(sel.index, np.intc):
                        for parm in effective_display_parms:
                            if parm in selected_df.columns:
                                value = selected_df[parm].iloc[sel.index]
                                add_parameter(parm, value)
                    elif isinstance(sel.index, np.float64):
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
                        print("Invalid index type for display parameters")
                        return
                    
                sel.annotation.set(text=annotation_text, color='black')
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        if return_axes:
            return axes

def marker_map(value):
    markers = ['o', 's', 'D',  'v', '^', '<', '>', 'd', 'H', 'p', '*']
    return markers[value % len(markers)]

# Example usage
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
