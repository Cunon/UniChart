
# UniChart

UniChart is a Python-based interactive plotting application using Tkinter and Matplotlib. This tool allows users to load datasets, apply various queries and formatting options, and visualize the data in a unified plot. 

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Launching the Application](#launching-the-application)
  - [Loading Datasets](#loading-datasets)
  - [Plotting Data](#plotting-data)
  - [Available Commands](#available-commands)
- [Example UCMD File](#example-ucmd-file)
- [Contributing](#contributing)
- [License](#license)

## Features
- Interactive plotting with Matplotlib
- Dataset management with filtering and formatting options
- Support for CSV, Excel, and custom UCMD files
- Command history and execution environment
- Dark mode support for plots

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unichart.git
   cd unichart
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Launching the Application
To launch the UniChart GUI, run the following command:
```bash
python unichart_gui.py
```

### Loading Datasets
You can load datasets into the application via the GUI menu or by executing commands.

#### Via GUI
- Go to `File -> Open` and select a CSV, Excel, or UCMD file to load.

#### Via Command
In the command entry box, you can use:
```python
load_df(pd.read_csv('path/to/yourfile.csv'))
```

### Plotting Data
To plot data, you need to specify the x and y columns. Use the command entry box to enter plotting commands.

```python
plot(x='column_x', y='column_y')
```

### Available Commands
Below are some of the key commands you can use in the command entry box:

- **Loading and Managing Datasets:**
  - `load_df(df, title=None, allcaps=True, load_cols_as_vars=True)`
  - `ucmd_file(file_path)`

- **Plotting:**
  - `plot(x, y, color=None, hue=None, marker=None, markersize=12, marker_edge_color='black', hue_palette=default_hue_palette, hue_order=None, line=False)`

- **Formatting:**
  - `color(uset_slice, color)`
  - `marker(uset_slice, marker)`
  - `linestyle(uset_slice, linestyle)`

- **Querying and Selecting Data:**
  - `query(uset_slice, query)`
  - `omit(uset_slice)`
  - `select(uset_slice)`
  - `restore(uset_slice)`

- **Utility:**
  - `print_columns(df)`
  - `print_usets()`
  - `clear()`
  - `save_png(filename)`
  - `save_ucmd(filename)`

### Example UCMD File
You can automate command execution using a UCMD file. Below is an example of a UCMD file (`example_ucmd_file.ucmd`):

```plaintext
load_df(pd.read_csv('data/sample.csv'))
query(uset[0], 'column_x > 50')
plot(x='column_x', y='column_y', color='blue')
```

To execute this file, use the command:
```python
ucmd_file('path/to/example_ucmd_file.ucmd')
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
