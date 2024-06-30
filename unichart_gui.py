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

class ReadOnlyFunction:
    """
    A class to wrap a function and make it read-only.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __setattr__(self, name, value):
        """
        Prevent setting attributes on the ReadOnlyFunction instance.
        """
        if name == 'func':
            super().__setattr__(name, value)
        else:
            raise AttributeError("Cannot modify read-only function")

class UniChart:
    """
    A class to represent the UniChart application for plotting datasets interactively using Tkinter.

    Attributes:
        root (tk.Tk): The root Tkinter window.
        command_history (list): A list to store command history.
        history_index (int): The index for navigating through command history.
        figure (Figure): The Matplotlib figure used for plotting.
        canvas (FigureCanvasTkAgg): The canvas for displaying the Matplotlib figure.
        toolbar (NavigationToolbar2Tk): The navigation toolbar for the Matplotlib figure.
        history (scrolledtext.ScrolledText): The text widget for displaying command history.
        entry (scrolledtext.ScrolledText): The text widget for entering commands.
        exec_env (dict): The execution environment for running commands.
        loaded_sets (list): A list of loaded datasets.
        last_x (str): The last used x-axis value.
        last_y (str): The last used y-axis value.
        dark_mode (bool): Flag to indicate if dark mode is enabled.
        suptitle (str): The title for the plot.
        display_parms (list): List of parameters to display.
        default_display_parms (list): List of parameters to pass with dfs when loaded.
    """

    def __init__(self, root):
        """
        Initialize the UniChart application.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("UniChart")

        # Configure the root window for high DPI displays
        self.root.tk.call('tk', 'scaling', 1.5)

        # Command history
        self.command_history = []
        self.history_index = -1

        # Create a figure for plotting
        self.figure = Figure(figsize=(10, 8), dpi=100)
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

        # Create an execution environment
        self.initialize_exec_env()

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

        # Initialize dark mode flag
        self.dark_mode = False

        # Bind Control-O to load_file
        self.root.bind("<Control-o>", lambda event: self.load_file())        # Execute startup script if it exists
        self.execute_startup_script()

    def initialize_exec_env(self):
        """
        Initialize the execution environment for running commands.
        """
        self.exec_env = {
            # Libraries
            'os': os,
            'plt': plt,
            'pd': pd,
            'sns': sns,
            'interp1d': interp1d,

            # Top level Plot formatting
            'display_parms': self.display_parms,
            'default_display_parms': [],
            'suptitle': self.suptitle,

            # Defaults
            'default_hue_palette': default_hue_palette,
            'table_read': table_read,
            'uniplot': uniplot,
            'marker_map': marker_map,
            'Dataset': Dataset,

            # Loaded Functions
            'plot': ReadOnlyFunction(self.plot),  

            # Selection and filtering
            'omit': ReadOnlyFunction(self.omit),
            'select': ReadOnlyFunction(self.select),
            'restore': ReadOnlyFunction(self.restore),
            'query': ReadOnlyFunction(self.query),

            # Set formatting
            'color': ReadOnlyFunction(self.color),  
            'marker': ReadOnlyFunction(self.marker),
            'linestyle': ReadOnlyFunction(self.linestyle),  

            # Data management
            'load_df': ReadOnlyFunction(self.load_df),
            'ucmd_file': ReadOnlyFunction(self.ucmd_file),
            'ucmdfile': ReadOnlyFunction(self.ucmd_file),
            'delta': ReadOnlyFunction(self.delta),

            # Utility functions
            'print_usets': ReadOnlyFunction(self.print_usets), 
            'list_usets': ReadOnlyFunction(self.print_usets), 
            'list_sets': ReadOnlyFunction(self.print_usets),

            'print_columns': print_columns,
            'list_parms': ReadOnlyFunction(self.print_columns_in_dataset),
            'list_cols': ReadOnlyFunction(self.print_columns_in_dataset),

            # Other functions
            'clear': ReadOnlyFunction(self.clear),
            'restart': ReadOnlyFunction(self.restart_program),
            'help': ReadOnlyFunction(self.help),
            'save_png': ReadOnlyFunction(self.save_png),
            'save_ucmd': ReadOnlyFunction(self.save_ucmd),
            'cd': ReadOnlyFunction(self.cd),
            'pwd': ReadOnlyFunction(self.pwd),
            'ls': ReadOnlyFunction(self.ls),

            'uset': [], #initialize empty list of datasets
            'toggle_darkmode': ReadOnlyFunction(self.toggled_darkmode)
        }

    def execute_startup_script(self):
        """
        Execute the startup.ucmd script if it exists in the same directory.
        """
        startup_file = os.path.join(os.path.dirname(__file__), "startup.ucmd")
        if os.path.isfile(startup_file):
            self.ucmd_file(startup_file)

    def cd(self, path):
        try:
            os.chdir(path)
            print(f"Changed directory to: {os.getcwd()}\n")
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

    def marker(self, uset_slice=None, marker=None):
        """
        Set the marker style for datasets.

        Args:
            uset_slice (list or Dataset, optional): The list of datasets or a single dataset to set marker. Default is None.
            marker (str): The marker style to set.
        """
        if marker is not None:
            uset_slice = self.get_uset_slice(uset_slice)
            for dataset in uset_slice:
                dataset.marker = marker
        else:
            print("Error: marker must be provided.")

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

    def plot(self, x=None, y=None, list_of_datasets=None, formatting_dict=None, color=None, hue=None,
             marker=None, markersize=12, marker_edge_color=None,
             hue_palette=default_hue_palette, hue_order=None, line=False, 
             ignore_list=[], suppress_msg=False, display_parms=None):
        """
        Plot the datasets on the specified x and y axes.

        Args:
            x (str, optional): The x-axis column name. Default is None.
            y (str, optional): The y-axis column name. Default is None.
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
        """
        if x is None:
            x = self.last_x
        if y is None:
            y = self.last_y

        if x is None or y is None:
            print("Error: x and y must be provided at least once.")
            return

        self.last_x = x
        self.last_y = y

        self.figure.clf()  # Clear the current figure
        ax = self.figure.add_subplot(111)  # Add a new subplot

        uset = self.exec_env['uset']
        suptitle = self.exec_env['suptitle']
        display_parms = self.exec_env['display_parms']

        uniplot(uset, x, y, return_axes=False, suptitle=suptitle, display_parms=display_parms, axes=ax, dark_mode=self.dark_mode)
        
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
        """
        Execute a command entered in the entry widget.

        Args:
            event (tk.Event): The event that triggered the command execution.
        """
        command = self.entry.get("1.0", tk.END).strip()
        if command:
            # Add command to history
            self.command_history.append(command)
            if len(self.command_history) > 1000:
                self.command_history.pop(0)
            self.history_index = -1

            self.history.configure(state='normal')
            self.history.insert(tk.END, f"> {command}\n")
            self.history.configure(state='disabled')
            self.history.see(tk.END)
            self.entry.delete("1.0", 'end-1c')

            # Execute the command in the plotting environment
            try:
                exec(command, {}, self.exec_env)
                self.canvas.draw()
            except Exception as e:
                self.history.configure(state='normal')
                self.history.insert(tk.END, f"Error: {e}\n")
                self.history.configure(state='disabled')
                self.history.see(tk.END)
        return 'break'

    def add_newline(self, event):
        """
        Add a newline in the entry widget.

        Args:
            event (tk.Event): The event that triggered adding a newline.

        Returns:
            str: 'break' to interrupt the default behavior.
        """
        self.entry.insert(tk.INSERT, '\n')
        return 'break'

    def ucmd_file(self, file_path):
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
        if not filename:
            filename = f'plot_{self.last_x}_{self.last_y}'
        elif filename.endswith('.png'):
            filename = filename[:-4]

        temp_file_name = filename
        i = 1
        while os.path.exists(f"{filename}.png"):
            filename = f"d{temp_file_name}_{i}"
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
        settings_menu.add_command(label="Toggle Dark Mode", command=self.toggled_darkmode)
        menubar.add_cascade(label="Settings", menu=settings_menu)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Compatible Files", ["*.csv", "*.xlsx", "*.ucmd"]), 
                    ("CSV files", "*.csv"), 
                    ("Excel files", "*.xlsx"), 
                    ("UCMD files", "*.ucmd"), 
                    ("All files", "*.*")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    self.load_df(df)
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                    self.load_df(df)
                elif file_path.endswith('.ucmd'):
                    self.ucmd_file(file_path)
                else:
                    messagebox.showerror("File Type Error", "Unsupported file type.")
                    return
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
        Save the history lines starting with '> ' to a text file without the '> ' prefix.

        Args:
            filename (str): The name of the file to save the history to.
        """
        if filename is None:
            filename = "default_ucmd_file.ucmd"

        temp_file_name = filename
        while os.path.exists(filename):
            filename = f"{temp_file_name}_{i}.ucmd"
            i += 1
            if i > 1000:
                print("Error: Could not save file.")
                return

        self.history.configure(state='normal')
        history_text = self.history.get("1.0", tk.END).splitlines()
        self.history.configure(state='disabled')

        with open(filename, 'w') as file:
            for line in history_text:
                if line.startswith("> "):
                    file.write(line[2:] + '\n')

    def restart_program(self):
        """
        Restart the execution environment.
        """
        self.initialize_exec_env()
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
        Print the datasets currently in the environment.
        """
        uset = self.exec_env['uset']
        # Define the maximum length for the title before breaking it into a new line
        max_title_length = 35

        # Adjust the header to allocate more space for the title
        print(f"{'Set':<8}{'Title':<40}{'Points':<10}{'Parms':<10}")
        print("=" * 70)  # Increase the total length to accommodate the longer title

        for i, dataset in enumerate(uset):
            title = dataset.get_title()
            points = len(dataset.df)
            parms = len(dataset.df.columns)

            # Split the title into multiple lines if it's too long
            if len(title) > max_title_length:
                # Break the title into chunks of max_title_length
                title_lines = [title[j:j+max_title_length] for j in range(0, len(title), max_title_length)]
            else:
                title_lines = [title]

            # Print the first line of the title with the dataset info
            print(f"{i:<8}{title_lines[0]:<40}{points:<10}{parms:<10}")

            # If there are additional lines, print them on new lines with spacing to align with the title column
            for additional_line in title_lines[1:]:
                print(f"{' ':<8}{additional_line:<40}{' ':<10}{' ':<10}")

    def toggled_darkmode(self):
        """
        Toggle the dark mode for the plots.
        """
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        self.plot()  # Re-plot to apply the new style


class TextRedirector:
    """
    A class to redirect stdout and stderr to a Tkinter Text widget.

    Attributes:
        widget (tk.Text): The Text widget to redirect output to.
        tag (str): The tag to use for the redirected output.
    """

    def __init__(self, widget, tag="stdout"):
        """
        Initialize the TextRedirector.

        Args:
            widget (tk.Text): The Text widget to redirect output to.
            tag (str): The tag to use for the redirected output. Default is "stdout".
        """
        self.widget = widget
        self.tag = tag

    def write(self, str):
        """
        Write a string to the Text widget.

        Args:
            str (str): The string to write.
        """
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.configure(state='disabled')
        self.widget.see(tk.END)

    def flush(self):
        """
        Flush the stream (no-op for this implementation).
        """
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = UniChart(root)
    root.mainloop()
