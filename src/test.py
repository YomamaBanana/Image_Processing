import os
import numpy as np
import PySimpleGUI as sg

def get_tree_data(parent, dirname):
    treedata = sg.TreeData()

    # https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Tree_Element.py#L26
    def add_files_in_folder(parent, dirname):

        files = os.listdir(dirname)
        for f in files:
            fullname = os.path.join(dirname, f)
            if os.path.isdir(fullname):
                treedata.Insert(parent, fullname, f, values=[])#, icon=folder_icon)
                add_files_in_folder(fullname, fullname)
            else:

                treedata.Insert(parent, fullname, f, values=[
                                os.stat(fullname).st_size])#, icon=file_icon)

    add_files_in_folder(parent, dirname)

    return treedata



if __name__ == "__main__":
    treedata = get_tree_data("", os.getcwd())

    # メニュー
    menu_def = [["File", ["Open Folder"]]]

    # レイアウト作成
    layout = [[sg.Menu(menu_def)],
            [sg.Tree(data=treedata,
                headings=[],
                auto_size_columns=True,
                num_rows=24,
                col0_width=20,
                key="-TREE-",
                show_expanded=False,
                enable_events=True), sg.Canvas(key="-CANVAS-")]]

    window = sg.Window("CSV Viewer", layout, finalize=True, element_justification="center", font="Monospace 8", resizable=False)
    
    
    
    while True:
        event, values = window.read()
        # print(event, values)

        if event is None:
            break
        
        elif event == "Open Folder":
            # 表示するフォルダを変更する
            starting_path = sg.popup_get_folder("Folder to display")
            treedata = get_tree_data("", starting_path)
            window["-TREE-"].update(values=treedata)