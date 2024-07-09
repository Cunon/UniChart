import tkinter as tk
from tkinter import Menu, Listbox, filedialog, messagebox, simpledialog
from pandastable import Table, TableModel
import pandas as pd
import sys

# Opt-in to the future behavior
# pd.set_option('future.no_silent_downcasting', True)

def launchDFV():
    # Get the main module
    main_module = sys.modules['__main__']
    global_vars = main_module.__dict__
    app = DataFrameManagerApp(global_vars=global_vars)
    app.mainloop()

class DataFrameManagerApp(tk.Tk):
    def __init__(self, global_vars, external_env=None):
        super().__init__()
        
        self.title("DataFrame Manager")
        self.geometry("800x600")
        
        self.dataframes = {}
        self.global_vars = global_vars
        
        if external_env:
            self.external_env = external_env
            self.populate_external_env_listbox()

        self._setup_menu()
        self._setup_widgets()
        self.populate_listbox()

    def _setup_menu(self):
        menubar = Menu(self)
        self.config(menu=menubar)
        
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load DataFrame", command=self.load_dataframe)
        file_menu.add_command(label="Save DataFrame", command=self.save_dataframe)
        file_menu.add_command(label="Exit", command=self.quit)
        
        options_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(label="Rename DataFrame", command=self.rename_dataframe)
        options_menu.add_command(label="New DataFrame", command=self.new_dataframe)
        options_menu.add_command(label="Clear Selection", command=self.clear_selection)

        operations_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Operations", menu=operations_menu)
        operations_menu.add_command(label="Merge DataFrames", command=self.merge_dataframes)
        operations_menu.add_command(label="Align DataFrames", command=self.align_dataframes)
        operations_menu.add_command(label="Sort DataFrame", command=self.sort_dataframe)
        operations_menu.add_command(label="Modify DataFrame", command=self.modify_dataframe)
        operations_menu.add_command(label="Filter DataFrame", command=self.filter_dataframe)
        
        about_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="About", menu=about_menu)

    def _setup_widgets(self):
        left_frame = tk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        label = tk.Label(left_frame, text="DataFrames")
        label.pack()
        
        self.listbox = Listbox(left_frame, selectmode=tk.MULTIPLE)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        
        table_frame = tk.Frame(self)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.table = Table(table_frame)
        self.table.show()

    def get_df_var_dict(self):
        df_type = type(pd.DataFrame())
        user_df_vars = {k: v for k, v in self.global_vars.items() if not k.startswith('_') and isinstance(v, df_type)}
        return user_df_vars

    def populate_listbox(self):
        df_var_dict = self.get_df_var_dict()
        for key in df_var_dict.keys():
            self.listbox.insert(tk.END, key)
            self.dataframes[key] = df_var_dict[key]

    def populate_external_env_listbox(self):
        df_type = type(pd.DataFrame())
        user_df_vars = {k: v for k, v in self.external_env.items() if not k.startswith('_') and isinstance(v, df_type)}
        for key in user_df_vars.keys():
            self.listbox.insert(tk.END, key)
            self.dataframes[key] = user_df_vars[key]
            
    def load_dataframe(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx *.xls")])
        if file_path:
            try:
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                df_name = f"DataFrame {len(self.dataframes) + 1}"
                self.dataframes[df_name] = df
                self.listbox.insert(tk.END, df_name)
                self.global_vars[df_name] = df  # Add to globals
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load DataFrame: {e}")

    def save_dataframe(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No DataFrame selected.")
            return
        
        df_name = self.listbox.get(selection[0])
        df = self.dataframes[df_name]
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx *.xls")])
        if file_path:
            try:
                df.to_csv(file_path, index=False) if file_path.endswith('.csv') else df.to_excel(file_path, index=False)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save DataFrame: {e}")

    def rename_dataframe(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No DataFrame selected.")
            return
        
        old_name = self.listbox.get(selection[0])
        new_name = simpledialog.askstring("Rename DataFrame", "Enter new name:")
        
        if new_name:
            self.dataframes[new_name] = self.dataframes.pop(old_name)
            self.listbox.delete(selection[0])
            self.listbox.insert(selection[0], new_name)
            self.global_vars[new_name] = self.global_vars.pop(old_name)  # Update globals

    def new_dataframe(self):
        columns = simpledialog.askstring("New DataFrame", "Enter column names separated by commas:")
        if columns:
            columns = [col.strip() for col in columns.split(",")]
            df = pd.DataFrame(columns=columns)
            df_name = f"DataFrame {len(self.dataframes) + 1}"
            self.dataframes[df_name] = df
            self.listbox.insert(tk.END, df_name)
            self.global_vars[df_name] = df  # Add to globals

    def clear_selection(self):
        self.listbox.selection_clear(0, tk.END)
        self.table.updateModel(TableModel(pd.DataFrame()))
        self.table.redraw()

    def on_select(self, event):
        selection = event.widget.curselection()
        if selection:
            idx_list = list(selection) #list of indexes from listbox
            df_names = [event.widget.get(df_index) for df_index in idx_list]
            df_list = [self.dataframes[df] for df in df_names]
            df = pd.concat(df_list, ignore_index=True)
            df = df.infer_objects(copy=False)  # Explicitly convert data types
            self.table.updateModel(TableModel(df))
            self.table.redraw()

    def merge_dataframes(self):
        df1_name = simpledialog.askstring("Merge DataFrames", "Enter first DataFrame name:")
        df2_name = simpledialog.askstring("Merge DataFrames", "Enter second DataFrame name:")
        how = simpledialog.askstring("Merge DataFrames", "Enter how to merge (left, right, outer, inner):", initialvalue="inner")
        if df1_name in self.dataframes and df2_name in self.dataframes:
            df1 = self.dataframes[df1_name]
            df2 = self.dataframes[df2_name]
            on = simpledialog.askstring("Merge DataFrames", "Enter the column name to merge on:")
            merged_df = pd.merge(df1, df2, how=how, on=on)
            merged_df_name = f"Merged DataFrame {len(self.dataframes) + 1}"
            self.dataframes[merged_df_name] = merged_df
            self.listbox.insert(tk.END, merged_df_name)
            self.global_vars[merged_df_name] = merged_df  # Add to globals
        else:
            messagebox.showerror("Error", "One or both DataFrames not found.")

    def align_dataframes(self):
        df1_name = simpledialog.askstring("Align DataFrames", "Enter first DataFrame name:")
        df2_name = simpledialog.askstring("Align DataFrames", "Enter second DataFrame name:")
        if df1_name in self.dataframes and df2_name in self.dataframes:
            df1 = self.dataframes[df1_name]
            df2 = self.dataframes[df2_name]
            aligned_df1, aligned_df2 = df1.align(df2, join='outer', axis=1)
            aligned_df1_name = f"Aligned {df1_name}"
            aligned_df2_name = f"Aligned {df2_name}"
            self.dataframes[aligned_df1_name] = aligned_df1
            self.dataframes[aligned_df2_name] = aligned_df2
            self.listbox.insert(tk.END, aligned_df1_name)
            self.listbox.insert(tk.END, aligned_df2_name)
            self.global_vars[aligned_df1_name] = aligned_df1  # Add to globals
            self.global_vars[aligned_df2_name] = aligned_df2  # Add to globals
        else:
            messagebox.showerror("Error", "One or both DataFrames not found.")

    def sort_dataframe(self):
        df_name = simpledialog.askstring("Sort DataFrame", "Enter DataFrame name:")
        if df_name in self.dataframes:
            df = self.dataframes[df_name]
            by = simpledialog.askstring("Sort DataFrame", "Enter column name to sort by:")
            ascending = simpledialog.askstring("Sort DataFrame", "Sort ascending? (yes or no):", initialvalue="yes") == "yes"
            sorted_df = df.sort_values(by=by, ascending=ascending)
            sorted_df_name = f"Sorted {df_name}"
            self.dataframes[sorted_df_name] = sorted_df
            self.listbox.insert(tk.END, sorted_df_name)
            self.global_vars[sorted_df_name] = sorted_df  # Add to globals
        else:
            messagebox.showerror("Error", "DataFrame not found.")

    def modify_dataframe(self):
        df_name = simpledialog.askstring("Modify DataFrame", "Enter DataFrame name:")
        if df_name in self.dataframes:
            df = self.dataframes[df_name]
            modification = simpledialog.askstring("Modify DataFrame", "Enter modification code:")
            try:
                exec(modification)
                modified_df_name = f"Modified {df_name}"
                self.dataframes[modified_df_name] = df
                self.listbox.insert(tk.END, modified_df_name)
                self.global_vars[modified_df_name] = df  # Add to globals
            except Exception as e:
                messagebox.showerror("Error", f"Failed to modify DataFrame: {e}")
        else:
            messagebox.showerror("Error", "DataFrame not found.")

    def filter_dataframe(self):
        df_selection = tk.Toplevel(self)
        df_selection.title("Select DataFrame to Filter")
        df_selection.geometry("300x200")

        lb = Listbox(df_selection)
        lb.pack(fill=tk.BOTH, expand=True)
        for df_name in self.dataframes.keys():
            lb.insert(tk.END, df_name)

        def on_select(event):
            selected_idx = lb.curselection()
            if selected_idx:
                df_name = lb.get(selected_idx[0])
                df = self.dataframes[df_name]
                query_examples = (
                    "Enter filter expression:\n\nExamples:\n"
                    "Example syntax:\n"
                    "df[(df['ALT'] > 500) & (df['SETNO'].isin([1,3,5,10]))]\n"
                )
                query = simpledialog.askstring("Filter DataFrame", query_examples)
                if query:
                    try:
                        # Evaluate the filtering expression
                        filtered_df = eval(query)
                        filtered_df_name = f"Filtered {df_name}"

                        preview = messagebox.askquestion("Preview Filtered DataFrame", f"Preview the filtered DataFrame with query '{query}'?")
                        if preview == 'yes':
                            preview_window = tk.Toplevel(self)
                            preview_window.title(f"Preview - {filtered_df_name}")
                            preview_table_frame = tk.Frame(preview_window)
                            preview_table_frame.pack(fill=tk.BOTH, expand=True)

                            preview_table = Table(preview_table_frame, dataframe=filtered_df)
                            preview_table.show()

                            confirm = messagebox.askquestion("Confirm Filtered DataFrame", "Confirm and save the filtered DataFrame?")
                            if confirm == 'yes':
                                self.dataframes[filtered_df_name] = filtered_df
                                self.listbox.insert(tk.END, filtered_df_name)
                                self.global_vars[filtered_df_name] = filtered_df  # Add to globals
                        else:
                            messagebox.showinfo("Filter Cancelled", "Filtering operation was cancelled.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to filter DataFrame: {e}")

        lb.bind("<<ListboxSelect>>", on_select)

if __name__ == "__main__":
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'X': [10, 20, 30], 'Y': [40, 50, 60]})
    
    app = DataFrameManagerApp(globals())
    app.mainloop()
