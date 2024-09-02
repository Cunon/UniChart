import os
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import sys
from DataFrameViewer import DataFrameManagerApp as DFV
import numpy as np
import mplcursors

# Import functions
from datasets_and_plot_functions import (
    uniplot, default_hue_palette, table_read, marker_map, Dataset
)

def print_columns(df):
    """
    Neatly print the columns of the DataFrame with their indices.

    Args:
        df (pd.DataFrame): The DataFrame whose columns need to be printed.
    """
    print(f"{'Index':<10}{'Column Name':<30}")
    print("=" * 40)
    for i, col in enumerate(df.columns):
        print(f"{i:<10}{col:<30}")

class ReadOnlyDict(dict):
    """
    A dictionary that prevents overwriting of specific read-only keys.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_only_keys = set()

    def make_read_only(self, key):
        self.read_only_keys.add(key)

    def __setitem__(self, key, value):
        if key in self.read_only_keys:
            raise KeyError(f"Cannot modify read-only key: {key}")
        super().__setitem__(key, value)


class UniChart:
    def __init__(self, root, figsize=(10, 8)):
        """
        Initialize the UniChart application.

        Args:
            root (tk.Tk): The root Tkinter window.
            figsize (tuple, optional): Initial size of the canvas. Default is (10, 8).
        """
        self.root = root
        self.root.title("UniChart")

        # Configure the root window for high DPI displays
        self.root.tk.call('tk', 'scaling', 1.5)

        # Command history
        self.command_history = []
        self.history_index = -1

        # Create a figure for plotting with a customizable size
        self.figure = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky='nsew')

        # Add the navigation toolbar
        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Create a Text widget for history
        self.history = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=10, state='disabled')
        self.history.grid(row=2, column=0, columnspan=2, sticky='nsew')

        # Create a ScrolledText widget for input
        self.entry = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=2)
        self.entry.grid(row=3, column=0, columnspan=2, sticky='nsew')
        self.entry.bind("<Return>", self.execute_command)
        self.entry.bind("<Shift-Return>", self.add_newline)
        self.entry.bind("<Up>", lambda event: self.navigate_history(event, 'up'))
        self.entry.bind("<Down>", lambda event: self.navigate_history(event, 'down'))

        # Configure grid weights to ensure proper resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)  # Toolbar row
        self.root.grid_rowconfigure(2, weight=1, minsize=100)  # History row
        self.root.grid_rowconfigure(3, weight=0)  # Entry row
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Initialize figure and canvas variables
        self.suptitle = None
        self.display_parms = []

        # Redirect stdout and stderr to the history box
        sys.stdout = TextRedirector(self.history, "stdout")
        sys.stderr = TextRedirector(self.history, "stderr")

        # Add a menu bar
        self.create_menu()

        # Set up data management
        self.loaded_sets = []

        # Initialize last used x and y values
        self.last_x = None
        self.last_y = None
        self.last_format = 'stack'

        self.legend = 'default'
        self.legend_ncols = 1

        # Initialize dark mode flag
        self.dark_mode = False

        # Create an execution environment
        self.initialize_exec_env()

        # Bind Control-O to load_file
        self.root.bind("<Control-o>", lambda event: self.load_file())        # Execute startup script if it exists
        self.execute_startup_script()

    def set_canvas_size(self, width, height):
        """
        Set the size of the canvas.

        Args:
            width (float): The width of the canvas in inches.
            height (float): The height of the canvas in inches.
        """
        self.figure.set_size_inches(width, height)
        self.canvas.draw()

    def get_exec_env(self):
        return self.exec_env
    
    def initialize_exec_env(self):
        """
        Initialize the execution environment for running commands.
        """
        self.exec_env = ReadOnlyDict({
            # Libraries
            'os': os,
            'plt': plt,
            'pd': pd,
            'sns': sns,
            'interp1d': interp1d,
            'sys': sys,
            'DFV': DFV,
            'ReadOnlyDict': ReadOnlyDict,

            # Top level Plot formatting
            'display_parms': self.display_parms,
            'legend': self.legend,
            'legend_ncols': self.legend_ncols,
            'default_display_parms': [],
            'suptitle': self.suptitle,

            # Defaults
            'default_hue_palette': default_hue_palette,
            'table_read': table_read,
            'uniplot': uniplot,
            'marker_map': marker_map,
            'Dataset': Dataset,

            # Loaded Functions
            'plot':self.plot,  

            # Selection and filtering
            'omit':self.omit,
            'select':self.select,
            'restore':self.restore,
            'query':self.query,

            # Set formatting
            'color':self.color,  
            'marker':self.marker,
            'markersize':self.markersize,
            'linestyle':self.linestyle,  
            'hue':self.hue,  
            'plot_type':self.plot_type,  
            'title':self.title,  

            # Data management
            'load_df':self.load_df,
            'fucmd':self.fast_ucmd,
            'fast_ucmd':self.fast_ucmd,
            'ucmd_file':self.ucmd_file,
            'ucmdfile':self.ucmd_file,
            'delta':self.delta,
            'exec_env':self.get_exec_env,
            'order':self.order,

            # Utility functions
            'print_usets':self.print_usets, 
            'list_usets':self.print_usets, 
            'list_sets':self.print_usets,

            'print_columns': print_columns,
            'list_parms':self.print_columns_in_dataset,
            'list_cols':self.print_columns_in_dataset,

            # Other functions
            'clear':self.clear,
            'restart':self.restart_program,
            'help':self.help,
            'save_png':self.save_png,
            'save_ucmd':self.save_ucmd,
            'cd':self.cd,
            'pwd':self.pwd,
            'ls':self.ls,
            'mkdir':self.mkdir,

            'uset': [], #initialize empty list of datasets
            'toggle_darkmode':self.toggle_darkmode,
            'darkmode': self.darkmode
        })

        # Make specific keys read-only
        for key in ['plot', 'omit', 'select', 'restore', 'query', 'color', 'marker', 'linestyle', 'load_df',
                    'ucmd_file', 'delta', 'print_usets', 'list_parms', 'clear', 'restart', 'help', 'save_png',
                    'save_ucmd', 'cd', 'pwd', 'ls', 'toggle_darkmode', 'darkmode', 'hue', 'exec_env', 'sys',
                    'plot_type', 'markersize', 'title', 'mkdir']:
            self.exec_env.make_read_only(key)

    def execute_startup_script(self):
        """
        Execute the startup.ucmd script if it exists in the same directory.
        """
            
        color_file = os.path.join(os.path.dirname(__file__), "css_colors.ucmd")
        if os.path.isfile(color_file):
            self.ucmd_file(color_file)

        startup_file = os.path.join(os.path.dirname(__file__), "startup.ucmd")
        if os.path.isfile(startup_file):
            self.ucmd_file(startup_file)

        self.clear()


    def cd(self, path):
        try:
            os.chdir(path)
            print(f"Changed directory to: {os.getcwd()}\n")
        except Exception as e:
            print(f"Error: {e}\n", "stderr")

    def mkdir(self, path):
        try:
            os.mkdir(path)
            print(f"New Directory made: {os.path.abspath(path)}\n")
        except Exception as e:
            print(f"Error: {e}\n", "stderr")

    def pwd(self):
        path_to_wd = os.path.abspath(os.getcwd())
        print(f"{path_to_wd}\n")
        return path_to_wd

    def ls(self):
        list_of_files = os.listdir()
        print(f"{list_of_files}\n")
        return list_of_files

    def clear(self):
        """
        Clear the command history.
        """
        self.history.configure(state='normal')
        self.history.delete('1.0', tk.END)
        self.history.configure(state='disabled')

    def get_uset_slice(self, uset_slice, return_indeicies=False):
        if uset_slice is None:
            return self.exec_env['uset']
        elif isinstance(uset_slice, list):
            if isinstance(uset_slice[0], int):
                return [self.exec_env['uset'][i] for i in uset_slice]
            return uset_slice
        elif isinstance(uset_slice, int):
            return [self.exec_env['uset'][uset_slice]]
        else:
            return [uset_slice]

    def omit(self, uset_slice=None):
        """
        Omit datasets from being selected for plotting.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to omit. Default is None.
        """
        uset_slice = self.get_uset_slice(uset_slice)
        for dataset in uset_slice:
            dataset.select = False

    def select(self, uset_slice=None):
        """
        Select datasets for plotting.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to select. Default is None.
        """
        for uset in self.exec_env['uset']:
            uset.select = False
        uset_slice = self.get_uset_slice(uset_slice)
        for dataset in uset_slice:
            dataset.select = True

    def restore(self, uset_slice=None):
        """
        Select datasets for plotting.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to select. Default is None.
        """

        if uset_slice == "all":
            for uset in self.exec_env['uset']:
                uset.select = True
                
        else:
            deactive_sets = []

            for uset in self.exec_env['uset']:
                if uset.select == False:
                    deactive_sets.append(uset.index)

            uset_slice = self.get_uset_slice(uset_slice)
            
            restore_sets = []
            for uset in uset_slice:
                restore_sets.append(uset.index)

            # Inner merge between deactive_sets and uset_slice's indices
            restored_indices = list(set(deactive_sets) & set(restore_sets))

            for index in restored_indices:
                self.exec_env['uset'][index].select = True

    def query(self, uset_slice=None, query=None):
        """
        Apply a query to filter datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to query. Default is None.
            query (str, optional): The query string to filter the datasets. Default is None.
        """
        uset_slice = self.get_uset_slice(uset_slice)
        for dataset in uset_slice:
            dataset.query = query

    def print_columns_in_dataset(self, uset_slice=None):
        """
        Print the columns of the dataset(s) in the specified slice.

        Args:
            uset_slice (int, list of int, Dataset, or list of Dataset, optional): The slice of datasets to print columns for. Default is None.
        """
        uset_slice = self.get_uset_slice(uset_slice)

        for dataset in uset_slice:
            print(f"Dataset {dataset.index}: {dataset.get_title()}")
            print_columns(dataset.df)
            print("\n")

    def color(self, uset_slice=None, color=None):
        """
        Set the color for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to color. Default is None.
            color (str): The color to set.
        """
        if color is not None:
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.color = color
        else:
            print("Error: color must be provided.")

    def plot_type(self, uset_slice=None, plot_type=None):
        """
        Set the plot_type for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to color. Default is None.
            plot_type (str): The plot_type to set.
        """
        if plot_type is not None:
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.plot_type = plot_type
        else:
            print("Error: plot_type must be provided.")

    def hue(self, uset_slice=None, hue=None):
        """
        Set the hue for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to color. Default is None.
            hue (str): The hue to set.
        """
        if hue is not None:
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.hue = hue
        else:
            print("Error: hue must be provided.")

    def marker(self, uset_slice=None, marker=False):
        """
        Set the marker style for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to set marker. Default is None.
            marker (str): The marker style to set.
        """
        if marker is not False:
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.marker = marker

    def order(self, uset_slice=None, order=False):
        """
        Set the marker style for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to set marker. Default is None.
            marker (str): The marker style to set.
        """
        if order is not False:
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.order = order

    def markersize(self, uset_slice=None, markersize=None):
        """
        Set the marker style for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to set markersize. Default is None.
            markersize (str): The markersize style to set.
        """
        if isinstance(markersize, int):
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.markersize = markersize
        else:
            print("Error: markersize must be an integer.")

    def linestyle(self, uset_slice=None, linestyle='solid'):
        """
        Set the linestyle for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to set linestyle. Default is None.
            linestyle (str): The linestyle to set. Default is 'solid'.
        """
        if linestyle != 'solid':
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.linestyle = linestyle
        else:
            print("Error: linestyle must be provided.")

    def title(self, uset_slice=None, title = None):
        uset_slice = self.get_uset_slice(uset_slice)
        if isinstance(title, str):
            for dataset in uset_slice:
                dataset.title = title
        else:
            print('Error, title must be a string')

    def plot(self, x=None, y=None, z=None, list_of_datasets=None, formatting_dict=None, color=None, hue=None,
            marker=None, markersize=12, marker_edge_color=None,
            hue_palette=default_hue_palette, hue_order=None, line=False, 
            suppress_msg=False, interactive=True, display_parms=None, legend=None, legend_ncols=None,
            format=None, figsize=None):
        """
        Plot the datasets on the specified x and y axes.

        Args:
            x (str, optional): The x-axis column name. Defaults to last used x value.
            y (str, optional): The y-axis column name. Default is None.
            z (str, optional): The z-axis column name. Default is None.
            list_of_datasets (list, optional): List of Dataset objects to plot. Default is None.
            formatting_dict (dict, optional): Dictionary of formatting options. Default is None.
            color (str, optional): The color for the plot. Default is None.
            hue (str, optional): The column name for hue differentiation. Default is None.
            marker (str, optional): The marker style for the plot. Default is None.
            markersize (int, optional): The size of the markers. Default is 12.
            marker_edge_color (str, optional): The edge color of the markers. Default is None.
            hue_palette (str, optional): The palette for hue differentiation. Default is default_hue_palette.
            hue_order (list, optional): The order of hue levels. Default is None.
            line (bool, optional): If True, plot as a line plot. Default is False.
            format (str, optional): Format for arranging subplots. Can be 'stack', 'std', or 'square'. Default is 'stack'.
        """
        if x is None:
            x = self.last_x
        if y is None:
            y = self.last_y
        if format is None:
            format = self.last_format
        
        self.last_x = x
        self.last_y = y
        self.last_format = format

        if figsize is not None:
            try:
                self.set_canvas_size(figsize[0],figsize[1])
            except Exception as e:
                print(f"Error: {e}")
        else:
            figsize=(10, 8)

        acceptable_legend_values = ['default', 'on', 'off']
        if legend is None:
            legend = self.exec_env['legend']
        elif legend not in acceptable_legend_values:
            print(f"Error: legend must be one of {acceptable_legend_values}")
            legend = 'default'

        if legend_ncols is None:
            legend_ncols = self.exec_env['legend_ncols']
            if isinstance(legend_ncols, int) or (int(legend_ncols) == legend_ncols):    
                legend_ncols = int(legend_ncols)
            else:
                print("Error: legend must be an integer or integer-like value.")
                legend_ncols = 1

        if list_of_datasets is None:
            list_of_datasets = uset = self.exec_env['uset']

        suptitle = self.exec_env['suptitle']
        display_parms = self.exec_env['display_parms']
        
        self.figure.clf()  # Clear the current figure

        if isinstance(y, list):
            num_plots = len(y)

            if format == 'stack':
                axs = self.figure.subplots(num_plots, 1, squeeze=False).flatten()
            elif format == 'std':
                axs = self.figure.subplots(1, num_plots, squeeze=False).flatten()
            elif format in ['sq', 'square']:
                ncols = int(np.ceil(np.sqrt(num_plots)))
                nrows = int(np.ceil(num_plots / ncols))
                axs = self.figure.subplots(nrows, ncols, squeeze=False).flatten()
            else:
                print("Error: format must be one of 'stack', 'std', or 'square'.")
                return

            for i, y_val in enumerate(y):
                ax = axs[i]
                uniplot(uset, x, y_val, 
                        return_axes=False, 
                        suptitle=suptitle, 
                        grid=True, 
                        display_parms=display_parms, 
                        axes=ax, 
                        dark_mode=self.dark_mode, 
                        interactive=interactive,
                        legend=legend,
                        legend_ncols=legend_ncols,
                        figsize=figsize)

                if interactive:
                    cursor = mplcursors.cursor(ax)

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
                        effective_display_parms = display_parms if display_parms else selected_dataset.display_parms

                        if effective_display_parms:
                            header = '\n{:<25} {:<5}'.format('Parameter', 'Value')
                            annotation_text += header
                            annotation_text += '\n' + '-'*35

                            def add_parameter(parm, value, interp=False):
                                value_str = f'{value:.2f}' if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
                                interp_str = ' (interp)' if interp else ''
                                
                                formatted_line = f'{parm:<20} {value_str:>10}{interp_str}'
                                
                                nonlocal annotation_text
                                annotation_text += '\n' + formatted_line

                            if isinstance(sel.index, np.intc):
                                for parm in effective_display_parms:
                                    if parm in selected_df.columns:
                                        value = selected_df[parm].iloc[sel.index]
                                        add_parameter(parm, value)
                            elif isinstance(sel.index, (int, float, np.integer, np.floating)):
                                try:
                                    float_index = float(sel.index)
                                    low_index = np.floor(float_index)
                                    high_index = np.ceil(float_index)
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

        else:
            ax = self.figure.add_subplot(111)  # Add a new subplot
            uniplot(uset, x, y, 
                    return_axes=False, 
                    suptitle=suptitle, 
                    grid=True, 
                    display_parms=display_parms, 
                    axes=ax, 
                    dark_mode=self.dark_mode, 
                    interactive=interactive,
                    legend=legend,
                    legend_ncols=legend_ncols,
                    figsize=figsize)

            if interactive:
                cursor = mplcursors.cursor(ax)

                @cursor.connect("add")
                def on_add(sel):
                    selected_title = sel.artist.get_label()
                    try:
                        set_number = int(selected_title.split(":")[0])
                    except ValueError:
                        print(f"Unexpected label format: {selected_title}")
                        return

                    selected_dataset = list_of_datasets[set_number]
                    selected_df = selected_dataset.df

                    annotation_text = f'Point: ({sel.target[0]:.2f}, {sel.target[1]:.2f})\nDataset {selected_dataset.index}: {selected_dataset.title}'
                    effective_display_parms = display_parms if display_parms else selected_dataset.display_parms

                    if effective_display_parms:
                        header = '\n{:<25} {:<5}'.format('Parameter', 'Value')
                        annotation_text += header
                        annotation_text += '\n' + '-'*35

                        def add_parameter(parm, value, interp=False):
                            value_str = f'{value:.2f}' if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
                            interp_str = ' (interp)' if interp else ''
                            
                            formatted_line = f'{parm:<20} {value_str:>10}{interp_str}'
                            
                            nonlocal annotation_text
                            annotation_text += '\n' + formatted_line

                        if isinstance(sel.index, np.intc):
                            for parm in effective_display_parms:
                                if parm in selected_df.columns:
                                    value = selected_df[parm].iloc[sel.index]
                                    add_parameter(parm, value)
                        elif isinstance(sel.index, (int, float, np.integer, np.floating)):
                            try:
                                float_index = float(sel.index)
                                low_index = np.floor(float_index)
                                high_index = np.ceil(float_index)
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

        self.canvas.draw()  # Update the canvas


    def load_df(self, df, title=None, allcaps=True, load_cols_as_vars=True):
        """
        Load a DataFrame into the environment as datasets.

        Args:
            df (pd.DataFrame): The DataFrame to load.
            title (str, optional): The title for the datasets. Default is None.
            allcaps (bool, optional): Whether to convert column names to uppercase. Default is True.
            load_cols_as_vars (bool, optional): Whether to load column names as variables. Default is True.
        """
        uset = self.exec_env['uset']
        next_index = len(uset)

        # If no title is given, look for a TITLE column in the DataFrame
        if not title: 
            if not "TITLE" in df.columns:
                df["TITLE"] = "Default Settitle"
                title = "Default Settitle"

        if allcaps: 
            df.columns = [col.upper() for col in df.columns]

        if load_cols_as_vars:
            for col in df.columns:
                col = col.replace(":", "")
                col = col.replace(".", "_")
                try:
                    exec(f"{col} = '{col}'", {}, self.exec_env)
                    exec(f"{col.lower()} = '{col}'", {}, self.exec_env)
                except Exception as e:
                    print(f"{e}: {col} not loaded as variable.")

        # Loop through unique titles and create subsets of the DataFrame
        display_parms = self.exec_env['default_display_parms']
        for title_col in df["TITLE"].unique():
            df_subset = df[df["TITLE"] == title_col]
            dataset = Dataset(df_subset, index=next_index, display_parms=display_parms)
            uset.append(dataset)
                        
            print(f"Set {next_index}: {dataset.get_title()}")

            next_index += 1

    def delta(self, base_set, new_set=None, delta_parms=None, align_on=None, suffixes=("", "NEW"), store_all_parms=True, passed_parms=None):
        #check types of base_set and new_set 
        uset = self.exec_env['uset']

        if delta_parms == None:
            print("Please provide parms to take deltas with")
            return False
        lsuffix = suffixes[0]
        rsuffix = suffixes[1]
        delta_type="point"
        if delta_type == "point": 
            df_base = base_set.df
            df_new = new_set.df

            merged_df = pd.merge(left=df_base, right=df_new, suffixes=suffixes, how="inner", on=align_on)

            for parm in delta_parms:
                try:
                    merged_df[f"DL{parm}"] = merged_df[f"{parm}{rsuffix}"] - merged_df[f"{parm}{lsuffix}"]
                    merged_df[f"DLPCT{parm}"] = 100 * (merged_df[f"{parm}{rsuffix}"] - merged_df[f"{parm}{lsuffix}"])/merged_df[f"{parm}{lsuffix}"]
                except:
                    print(f"Couldn't take delta for {parm}")

            if len(merged_df) > 0:
                self.load_df(merged_df)
                uset[-1].settype = 'delta'
                uset[-1].delta_sets = (base_set.index, new_set.index)
                uset[-1].title = f"Delta set {base_set.index}-{new_set.index}"

    def execute_command(self, event):
        command = self.entry.get("1.0", tk.END).strip()
        if command:
            self.command_history.append(command)
            if len(self.command_history) > 1000:
                self.command_history.pop(0)
            self.history_index = -1

            self.history.configure(state='normal')
            self.history.insert(tk.END, f"> {command}\n", "stdcmd")  # Use 'stderr' for bold and red
            self.history.configure(state='disabled')
            self.entry.delete("1.0", 'end-1c')

            try:
                exec(command, {}, self.exec_env)
                self.canvas.draw()
            except Exception as e:
                self.history.configure(state='normal')
                self.history.insert(tk.END, f"Error: {e}\n", "stderr")
                self.history.configure(state='disabled')
            self.history.see(tk.END)
        return 'break'


    def add_newline(self, event):
        """
        Add a newline in the entry widget.
        """
        self.entry.insert(tk.INSERT, '\n')
        return 'break'


    def fast_ucmd(self, file_path):
        """
        Execute commands from a UCMD file.

        Args:
            file_path (str): The path to the UCMD file.
        """
        try:
            with open(file_path, 'r') as file:
                commands = file.read()  # Read the entire file content

            command_history = "\n".join([f"> {line}" for line in commands.splitlines()])

            self.history.configure(state='normal')
            self.history.insert(tk.END, f"{command_history}\n")
            self.history.configure(state='disabled')
            self.history.see(tk.END)
            
            try:
                exec(commands, {}, self.exec_env)
            except Exception as e:
                self.history.configure(state='normal')
                self.history.insert(tk.END, f"Error: {e}\n")
                self.history.configure(state='disabled')
                self.history.see(tk.END)
            
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("File Execution Error", f"Could not execute file: {e}")


    def ucmd_file(self, file_path):
        """
        Execute commands from a UCMD file, handling indented blocks, multi-line commands, and syntax errors.

        Args:
            file_path (str): The path to the UCMD file.
        """
        try:
            line_number=0
            with open(file_path, 'r') as file:
                lines = file.readlines()

            command_block = []
            inside_multiline = False
            block_indent = None

            def execute_and_reset_block():
                if command_block:
                    self.execute_command_block(command_block)
                    command_block.clear()

            for line_number, line in enumerate(lines, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue  # Skip empty lines

                current_indent = len(line) - len(stripped_line)

                # Handle multiline structures
                if inside_multiline:
                    command_block.append(line)
                    # Detect the end of multiline structure
                    if stripped_line.endswith((']', '}', ')')):
                        inside_multiline = False
                        execute_and_reset_block()
                    continue

                # Detect the start of a multiline structure
                if stripped_line.endswith(('[', '{', '(')):
                    inside_multiline = True
                    command_block.append(line)
                    continue

                if block_indent is None:
                    block_indent = current_indent  # Start of a new block

                # Append line to the current block
                if current_indent >= block_indent:
                    command_block.append(line)
                else:
                    # We have dedented, meaning the end of the current block
                    execute_and_reset_block()
                    block_indent = current_indent
                    command_block.append(line)

            # Execute any remaining commands in the block
            execute_and_reset_block()

        except Exception as e:
            error_message = f"Error executing file '{file_path}' on line {line_number}: {str(e)}"
            messagebox.showerror("File Execution Error", error_message)
            print(error_message, file=sys.stderr)


    def execute_command_block(self, command_block):
        """
        Execute a multi line block of commands.
        """
        try:
            commands = "\n".join(command_block)
            self.history.configure(state='normal')
            
            # Iterate over each line in the command block to apply styling
            for line in command_block:
                command = line.strip()
                if command:
                    # Check for comments within the command
                    comment_index = command.find('#')
                    if comment_index != -1:
                        # Split the command at the comment and insert each part with appropriate tags
                        self.history.insert(tk.END, f"> {command[:comment_index]}", "stdcmd") 
                        self.history.insert(tk.END, command[comment_index:], "comment")  
                        self.history.insert(tk.END, "\n", "comment") 
                    else:
                        self.history.insert(tk.END, f"> {command}\n", "stdcmd")
            self.history.configure(state='disabled')
            self.history.see(tk.END)
            
            # Execute the commands within the local execution environment
            exec(commands, {}, self.exec_env)
            self.canvas.draw()
        except Exception as e:
            self.history.configure(state='normal')
            self.history.insert(tk.END, f"Error: {e}\n", "stderr")
            self.history.configure(state='disabled')
            self.history.see(tk.END)

            
    def navigate_history(self, event, direction):
        """
        Navigate through the command history.

        Args:
            event (tk.Event): The event that triggered history navigation.
            direction (str): The direction to navigate ('up' or 'down').
        """
        if self.command_history:
            if direction == 'up':
                if self.history_index == -1:
                    self.history_index = len(self.command_history) - 1
                elif self.history_index > 0:
                    self.history_index -= 1
            elif direction == 'down':
                if self.history_index == -1:
                    return
                elif self.history_index < len(self.command_history) - 1:
                    self.history_index += 1
                else:
                    self.history_index = -1
                    self.entry.delete("1.0", tk.END)
                    return
            self.entry.delete("1.0", "end-1c")
            self.entry.insert(tk.END, self.command_history[self.history_index])
            
            # Error, cursor not moving to the end
            self.entry.mark_set(tk.INSERT, tk.END)  # Move cursor to the end
            self.entry.mark_set(tk.INSERT, "end-1c")  # Move cursor to the end

    def save_png(self, filename=None):
        """
        Save the current plot as a PNG file.

        Args:
            filename (str, optional): The filename to save the plot as. Default is None.
        """
        x = self.last_x
        y = self.last_y

        if isinstance(y,list):
            y = "_".join(y)

        if not filename:
            filename = f'plot_{x}_vs_{y}'
        elif filename.endswith('.png'):
            filename = filename[:-4]

        temp_file_name = filename
        i = 1
        while os.path.exists(f"{filename}.png"):
            filename = f"{temp_file_name}_{i}"
            i += 1
            if i > 1000:
                print("Error: Could not save file.")
                return False
            
        filename += ".png"
        self.figure.savefig(filename)
        print(f"Plot saved as {filename}")

    def help(self, function=None):
        """
        Display help information for a function or list available functions.

        Args:
            function (str, optional): The function name to display help for. Default is None.
        """
        
        uset = self.exec_env['uset']

        if (function == 'uset') or (function == uset):
            print(uset[0].__doc__)
        else:
            libraries = {
                'os': 'Builtin os library.',
                'plt': 'Matplotlib.pyplot',
                'pd': 'Pandas',
                'sns': 'Seaborn',
                'interp1d': 'Scipy.interpolate.interp1d',
            }

            defaults = {
                'default_hue_palette': 'Hue palette UniChart uses for coloring datasets.',
                'marker_map': 'List of markers in order that UniChart assigns to datasets.',
            }

            builtin_functions = {
                'plot': 'plot(x,y) - Plot the datasets in the environment on the x and y axes.',
                'load_df': 'load_df(df) - Load a DataFrame into the environment as a dataset.',
                'ucmd_file': 'ucmd_file(file_path) - Execute a ucmd file in the environment.',
                'print_usets': 'print_usets() - Print the datasets in the environment.',
                'print_columns': 'print_columns(df) - Print the columns of a DataFrame.',
                'list_parms': 'list_parms(uset_slice) - List the parameters in a dataset slice.',
                'list_cols': 'list_cols(uset_slice) - List the columns in a dataset slice.',
                'clear': 'clear() - Clear the history box.',
                'restart': 'restart() - Restart the environment.',
                'save_png': 'save_png(filename) - Save the current plot as a PNG file',
                'save_ucmd': 'save_ucmd(filename) - Save the command history to a UCMD file.',
                'cd': 'cd(path) - Change the current directory.',
                'pwd': 'pwd() - Print the current working directory.',
                'ls': 'ls() - List files in the current directory.',
                'uset': 'list of datasets in the environment',
                'toggle_darkmode': 'toggle_darkmode() - Toggle dark mode for plots.',
                'delta': 'delta(base_set, new_set, delta_parms, align_on, suffixes, store_all_parms, passed_parms) - Compute deltas between datasets.'
            }

            selection_functions = {
                'omit': 'omit(uset_slice) - Omit datasets from being selected for plotting.',
                'select': 'select(uset_slice) - Select datasets for plotting.',
                'restore': 'restore(uset_slice) - Restore previously omitted datasets.',
                'query': 'query(uset_slice, query) - Apply a query to filter datasets.',
            }

            format_functions = {
                'color': 'color(uset_slice, color) - Set the color of datasets.',
                'marker': 'marker(uset_slice, marker) - Set the marker of datasets.',
                'linestyle': 'linestyle(uset_slice, linestyle) - Set the linestyle of datasets.',
                'hue': 'hue(uset_slice, hue) - Set hue differentiation for datasets.',
                'plot_type': 'plot_type(uset_slice, plot_type) - Set the plot type for datasets.',
                'markersize': 'markersize(uset_slice, markersize) - Set the marker size for datasets.',
                'order': 'order(uset_slice, order) - Set the order for datasets.',
            }

            max_len_lib = max(len(key) for key in libraries.keys())
            max_len_def = max(len(key) for key in defaults.keys())
            max_len_func = max(len(key) for key in builtin_functions.keys())
            max_len_form = max(len(key) for key in format_functions.keys())
            max_len_sel = max(len(key) for key in selection_functions.keys())

            print("Default Libraries:\n")
            for key, description in libraries.items():
                print(f"{key:<{max_len_lib}} : {description}")

            print("\nFunctions:\n")
            for key, description in builtin_functions.items():
                print(f"{key:<{max_len_func}} : {description}")

            print("\nDefault Attributes:\n")
            for key, description in defaults.items():
                print(f"{key:<{max_len_def}} : {description}")

            print("\nSelection functions:\n")
            for key, description in selection_functions.items():
                print(f"{key:<{max_len_sel}} : {description}")

            print("\nFormmating functions:\n")
            for key, description in format_functions.items():
                print(f"{key:<{max_len_form}} : {description}")

    def create_menu(self):
        """
        Create the menu bar for the application.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.load_file)
        file_menu.add_command(label="Save Png", command=self.save_file_dialog)
        file_menu.add_command(label="Execute ucmd File", command=self.load_ucmd_file)
        file_menu.add_command(label="Restart", command=self.restart_program)
        file_menu.add_command(label="Exit", command=self.exit_app)
        menubar.add_cascade(label="File", menu=file_menu)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="About", menu=about_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Toggle Dark Mode", command=self.toggle_darkmode)
        menubar.add_cascade(label="Settings", menu=settings_menu)

    def load_file(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Compatible Files", ["*.csv", "*.xlsx", "*.ucmd"]), 
                    ("CSV files", "*.csv"), 
                    ("Excel files", "*.xlsx"), 
                    ("UCMD files", "*.ucmd"), 
                    ("All files", "*.*")]
        )
        if file_paths:
            for file_path in file_paths:
                if file_path:
                    try:
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                            command = f"load_df(pd.read_csv(r'{file_path}'))"
                        elif file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                            command = f"load_df(pd.read_excel(r'{file_path}'))"
                        elif file_path.endswith('.ucmd'):
                            command =f"ucmd_file(r'{file_path}')"
                            self.ucmd_file(file_path)
                            df=None
                        else:
                            messagebox.showerror("File Type Error", "Unsupported file type.")
                            command = "\n"
                            return
                        
                        self.command_history.append(command)
                        print(f"> {command}")
                        if df is not None:
                            self.load_df(df)

                        print(f"Loaded {file_path}")
                    except Exception as e:
                        messagebox.showerror("File Load Error", f"Could not load file: {e}")

    def load_ucmd_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("UCMD files", "*.ucmd"), ("All files", "*.*")]
        )
        if file_path:
            self.ucmd_file(file_path)

    def save_file_dialog(self):
        """
        Save the current plot as a PNG file through a file dialog.
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            self.save_png(file_path)

    def save_ucmd(self, filename=None):
        """
        Save the command history to a text file, only including commands issued after the last 'restart()' command.

        Args:
            filename (str): The name of the file to save the command history to.
        """
        if filename is None:
            filename = "default_ucmd_file.ucmd"

        i = 0
        temp_file_name = filename
        while os.path.exists(filename):
            filename = f"{temp_file_name}_{i}.ucmd"
            i += 1
            if i > 1000:
                print("Error: Could not save file.")
                return

        last_restart_index = -1
        for index, command in enumerate(self.command_history):
            if 'restart()' in command.strip():
                last_restart_index = index

        with open(filename, 'w') as file:
            for command in self.command_history[last_restart_index+1:]:
                if "save_ucmd" not in command.strip():
                    file.write(command + '\n')

        print(f"Command history saved to {filename}")


    def restart_program(self):
        """
        Restart the execution environment.
        """
        self.initialize_exec_env()
        self.execute_startup_script()
        print("Environment restarted")

    def exit_app(self):
        """
        Exit the application.
        """
        self.root.quit()

    def show_about(self):
        """
        Show the 'About' information of the application.
        """
        messagebox.showinfo("About", "UniChart\nVersion 1.0\n\nA simple interactive plotting application.")

    def print_usets(self):
        """
        Print the datasets currently in the environment with additional details.
        """
        uset = self.exec_env['uset']
        # Define the maximum length for the title before breaking it into a new line
        max_title_length = 35

        # Adjust the header to allocate more space for the title and add 'Selected' and 'Query' columns
        print(f"{'Set':<6}{'Selected':<10}{'Title':<40}{'Points':<10}{'Parms':<10}{'Query':<30}")
        print("=" * 106)  # Increase the total length to accommodate the new columns

        for i, dataset in enumerate(uset):
            selected = 'Yes' if dataset.select else 'No'
            title = dataset.get_title()
            points = len(dataset.df)
            parms = len(dataset.df.columns)
            query = dataset.query if dataset.query else 'None'

            # Split the title into multiple lines if it's too long
            if len(title) > max_title_length:
                # Break the title into chunks of max_title_length
                title_lines = [title[j:j + max_title_length] for j in range(0, len(title), max_title_length)]
            else:
                title_lines = [title]

            # Print the first line of the title with the dataset info
            print(f"{i:<6}{selected:<10}{title_lines[0]:<40}{points:<10}{parms:<10}{query:<30}")

            # If there are additional lines, print them on new lines with spacing to align with the title column
            for additional_line in title_lines[1:]:
                print(f"{' ':<16}{additional_line:<40}{' ':<10}{' ':<10}{' ':<30}")

    def toggle_darkmode(self):
        """
        Toggle the dark mode for the plots.
        """
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        self.plot()  # Re-plot to apply the new style

    def darkmode(self):
        """
        Toggle the dark mode for the plots.
        """
        self.dark_mode = True
        self.plot()  # Re-plot to apply the new style

class TextRedirector:
    # Format text in textbox widget based on context

    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
        # Configure tags for different output styles
        self.widget.tag_configure("stdout", foreground="black")
        self.widget.tag_configure("stdcmd", foreground="black", font='TkDefaultFont 9 bold')
        self.widget.tag_configure("stderr", foreground="red")
        self.widget.tag_configure("comment", foreground="green", font="TkDefaultFont 9 italic")

    def write(self, str):
        self.widget.configure(state='normal')
        if self.tag == "stdout":
            # Check each line for comments
            for line in str.splitlines(True):
                comment_index = line.find('#')
                if comment_index != -1:
                    # Split the line at the comment and insert each part with appropriate tags
                    self.widget.insert(tk.END, line[:comment_index], "stdout")
                    self.widget.insert(tk.END, line[comment_index:], "comment")
                else:
                    self.widget.insert(tk.END, line, "stdout")
        else:
            self.widget.insert(tk.END, str, self.tag)
        self.widget.configure(state='disabled')
        self.widget.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniChart(root, figsize=(8,8))
    root.mainloop()
