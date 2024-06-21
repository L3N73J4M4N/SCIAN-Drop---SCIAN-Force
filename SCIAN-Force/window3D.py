from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from forces import Forces3D
import numpy as np
import pandas as pd
from PIL import ImageTk
from icons import icon_force_karina


class Inter3D:
    def __init__(self):
        self.obj = None  # objeto con el que se trabajará
        win = Tk()  # ventana
        x_win, y_win = 370, 550
        win.minsize(x_win, y_win)  # tamaño de la ventana
        win.title("SCIAN-Force (OFD 2024 Beta version)")  # título de la ventana
        icon16, icon32 = icon_force_karina()
        icon16, icon32 = ImageTk.PhotoImage(icon16), ImageTk.PhotoImage(icon32)
        win.iconphoto(False, icon32, icon16)
        x_screen, y_screen = win.winfo_screenwidth(), win.winfo_screenheight()
        x = x_screen // 2 - x_win // 2
        y = y_screen // 2 - y_win // 2
        win.geometry(str(x_win) + 'x' + str(y_win) + '+' + str(x) + '+' + str(y))
        win.resizable(None, None)

        def entry_data():
            value1.delete(0, 'end')
            value2.delete(0, 'end')
            value3.delete(0, 'end')
            value4.delete(0, 'end')
            mean = np.mean(self.obj.meanCurvature)
            std = np.std(self.obj.meanCurvature)
            maxi = np.max(self.obj.meanCurvature)
            mini = np.min(self.obj.meanCurvature)
            value1.insert(0, str(np.round(mean, 10)))
            value2.insert(0, str(np.round(std, 10)))
            value3.insert(0, str(np.round(maxi, 10)))
            value4.insert(0, str(np.round(mini, 10)))

        def openfile():
            filepath = filedialog.askopenfilename(initialdir='Desktop',
                                                  title='Select .OFF / .OBJ file',
                                                  filetypes=(('object files', '*.obj *.off'),
                                                             ('obj files', '*.obj'),
                                                             ('off files', '*.off'),
                                                             ('all files', '*.*')))
            if len(filepath) > 0:
                try:
                    self.obj = Forces3D(filepath, interval=interval.get())
                except ValueError:
                    self.obj = Forces3D(filepath)
                self.obj.fix_mesh()
                string = interval.get().split('x')
                string = '(' + string[0] + ' x ' + string[1] + ' x ' + string[2] + ')'
                label_down.configure(text=string)
                name = self.obj.name
                if len(self.obj.name) > 30:
                    name = self.obj.name[0:30] + '\n' + self.obj.name[29:]
                if len(self.obj.name) > 60:
                    name = self.obj.name[0:30] + '\n' + self.obj.name[29:60] + '\n' + self.obj.name[59:]
                obj_name.configure(text=name)
                radius = str(self.obj.get_radius())
                volume = str(self.obj.get_volume())
                if len(radius) >= 10:
                    radius = radius[0:10]
                if len(volume) >= 10:
                    volume = volume[0:10]
                value5.delete(0, 'end')
                value6.delete(0, 'end')
                value5.insert(0, radius)
                value6.insert(0, volume)

        def laplace():
            self.obj.laplacian_curvature()
            try:
                max_value = float(maxvalue.get())
            except ValueError:
                max_value = None
            try:
                min_value = float(minvalue.get())
            except ValueError:
                min_value = None
            self.gamma = 1 if gamma.get() == '' else float(gamma.get())
            self.obj.stress(self.gamma)
            self.obj.plot_go(max_value, min_value)
            # self.obj.colormap(normalize=normal.get(), max_value=max_value)
            # self.obj.plot(use_fun=True, view_method=True)
            entry_data()

        def discrete():
            self.obj.discrete_mean_curvature()
            try:
                max_value = float(maxvalue.get())
            except ValueError:
                max_value = None
            try:
                min_value = float(minvalue.get())
            except ValueError:
                min_value = None
            self.gamma = 1 if gamma.get() == '' else float(gamma.get())
            self.obj.stress(self.gamma)
            self.obj.plot_go(max_value, min_value)
            # self.obj.colormap(normalize=normal.get(), max_value=max_value)
            # self.obj.plot(use_fun=True, view_method=True)
            entry_data()

        def principal():
            self.obj.taubin_method()
            try:
                max_value = float(maxvalue.get())
            except ValueError:
                max_value = None
            try:
                min_value = float(minvalue.get())
            except ValueError:
                min_value = None
            self.gamma = 1 if gamma.get() == '' else float(gamma.get())
            self.obj.stress(self.gamma)
            self.obj.plot_go(max_value, min_value)
            # self.obj.colormap(normalize=normal.get(), max_value=max_value)
            # self.obj.plot(use_fun=True, view_method=True)
            entry_data()

        def tri():
            self.obj.trimesh_method()
            try:
                max_value = float(maxvalue.get())
            except ValueError:
                max_value = None
            try:
                min_value = float(minvalue.get())
            except ValueError:
                min_value = None
            self.gamma = 1 if gamma.get() == '' else float(gamma.get())
            self.obj.stress(self.gamma)
            self.obj.plot_go(max_value, min_value)
            # self.obj.colormap(normalize=normal.get(), max_value=max_value)
            # self.obj.plot(use_fun=True, view_method=True)
            entry_data()

        def rus():
            self.obj.rusinkiewicz_curvature()
            try:
                max_value = float(maxvalue.get())
            except ValueError:
                max_value = None
            try:
                min_value = float(minvalue.get())
            except ValueError:
                min_value = None
            self.gamma = 1 if gamma.get() == '' else float(gamma.get())
            self.obj.stress(self.gamma)
            self.obj.plot_go(max_value, min_value)
            # self.obj.colormap(normalize=normal.get(), max_value=max_value)
            # self.obj.plot(use_fun=True, view_method=True)
            entry_data()

        # object
        select_obj = LabelFrame(win, text='Object')
        select_obj.pack(pady=2)
        button_select = Button(select_obj, text='Select object file', command=openfile)
        obj_name = Label(select_obj, text='')
        interval = Entry(select_obj)
        interval.insert(0, '1x1x1')
        label_interval = Label(select_obj, text='Voxel size [μm³]:')
        label_down = Label(select_obj, text='(width x height x depth)')
        obj_name.grid(row=3, columnspan=2)
        interval.grid(row=0, column=1, padx=10)
        label_interval.grid(row=0, column=0)
        label_down.grid(row=1, columnspan=2)
        label_gamma = Label(select_obj, text='γ [mN/m]:')
        gamma = Entry(select_obj)
        label_gamma.grid(row=4, column=0)
        gamma.grid(row=4, column=1)

        # colorbar
        cb = LabelFrame(win, text='Colorbar')
        cb.pack(pady=2)
        maxvalue_label = Label(cb, text='Max value:')
        minvalue_label = Label(cb, text='Min value:')
        maxvalue = Entry(cb, width=15)
        minvalue = Entry(cb, width=15)
        maxvalue_label.grid(column=1, row=0)
        minvalue_label.grid(column=1, row=1)
        maxvalue.grid(column=2, row=0, padx=20)
        minvalue.grid(column=2, row=1)

        # plot
        plot = LabelFrame(win, text='Plot')
        plot.pack(pady=2)
        button_discrete = Button(plot, text='Discrete method', command=discrete)
        button_laplace = Button(plot, text='Laplacian method', command=laplace)
        button_principal = Button(plot, text='Taubin method', command=principal)
        button_tri = Button(plot, text='Trimesh method', command=tri)
        button_rus = Button(plot, text='Rusinkiewicz method', command=rus)

        # pack de los botones
        button_select.grid(row=2, columnspan=2)
        button_discrete.pack(padx=50)
        button_laplace.pack()
        button_principal.pack()
        button_tri.pack()
        button_rus.pack()

        # valores
        resres = LabelFrame(win, text='Results of droplet geometry')
        res5 = Label(resres, text='radius [μm]:')
        value5 = Entry(resres)
        res6 = Label(resres, text='volume [μm³]:')
        value6 = Entry(resres)
        res = LabelFrame(win, text='Results of stress measurement')
        res1 = Label(res, text='mean:')
        value1 = Entry(res)
        res2 = Label(res, text='std:')
        value2 = Entry(res)
        res3 = Label(res, text='max:')
        value3 = Entry(res)
        res4 = Label(res, text='min:')
        value4 = Entry(res)

        # guardar los resultados de las curvaturas
        def save():
            dicty = {'vertex': [], 'coord': [], 'normal_stress': []}
            for i in range(0, len(self.obj.meanCurvature)):
                dicty['vertex'].append(i)
                coord = self.obj.mesh[0][i]
                dicty['coord'].append((coord[0], coord[1], coord[2]))
                dicty['normal_stress'].append(self.obj.meanCurvature[i])
            df = pd.DataFrame(dicty)
            name_path = filedialog.asksaveasfilename(initialdir="Desktop",
                                                     initialfile=self.obj.name[0:-4] + '-' + self.obj.method[19:],
                                                     defaultextension='.xlsx',
                                                     title="Save as",
                                                     filetypes=(("xlsx files", "*.xlsx"),
                                                                ("all files", "*.*")))
            if len(name_path) > 0:
                if name_path[-5::] != '.xlsx':
                    name_path += '.xlsx'
                df.to_excel(name_path, sheet_name='results', index=False)

        save_curvatures = Button(res, text='Save stresses', command=save)

        resres.pack(pady=2)
        res5.grid(column=0, row=4)
        res6.grid(column=0, row=5)
        value5.grid(column=1, row=4)
        value6.grid(column=1, row=5)

        res.pack(pady=2)
        res1.grid(column=0, row=0)
        res2.grid(column=0, row=1)
        res3.grid(column=0, row=2)
        res4.grid(column=0, row=3)

        value1.grid(column=1, row=0, padx=17)
        value2.grid(column=1, row=1)
        value3.grid(column=1, row=2)
        value4.grid(column=1, row=3)
        save_curvatures.grid(row=6, columnspan=2)

        # iniciar ventana
        win.mainloop()


Inter3D()
