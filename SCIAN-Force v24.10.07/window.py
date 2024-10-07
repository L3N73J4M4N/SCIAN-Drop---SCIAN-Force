from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import numpy as np
from skimage import io
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from curves import get_splines, get_curvatures
from skimage.util import img_as_ubyte
from colors import get_scale, get_subscale
import os
from matplotlib.colorbar import ColorbarBase
import cv2 as cv
from shapely import Polygon
from icons import icon_force_karina
from PIL import ImageTk


class Forces2D:
    def __init__(self):
        win = Tk()
        win.title('SCIAN-Force 2D (v.2024.06.07)')
        x_screen, y_screen = win.winfo_screenwidth(), win.winfo_screenheight()
        x = x_screen // 2 - 250
        y = y_screen // 2 - 285
        win.geometry('500x550+' + str(x) + '+' + str(y))
        win.resizable(None, None)
        icon16, icon32 = icon_force_karina()
        icon16, icon32 = ImageTk.PhotoImage(icon16), ImageTk.PhotoImage(icon32)
        win.iconphoto(False, icon32, icon16)
        self.stack = []  # stack de imágenes
        self.curves = []  # curvaturas de cada imagen
        self.minmax = []  # mínimo y máximo
        self.i = []  # índices
        self.colormap = None  # escala de colores general
        self.mean = []
        self.k0 = None

        def window_info():
            sub_win = Toplevel()
            sub_win.title('Information')
            x_screen, y_screen = win.winfo_screenwidth(), win.winfo_screenheight()
            x = x_screen // 2 - 150
            y = y_screen // 2 - 75
            sub_win.geometry('300x150+' + str(x) + '+' + str(y))
            sub_win.resizable(None, None)
            icon16, icon32 = icon_force_karina()
            icon16, icon32 = ImageTk.PhotoImage(icon16), ImageTk.PhotoImage(icon32)
            sub_win.iconphoto(False, icon32, icon16)
            z = int(slice_z.get())
            info = Label(sub_win, text='max(k) = ' + str(self.minmax[z][1])
                         + '\nmin(k) = ' + str(self.minmax[z][0])
                         + '\nmean(k) = ' + str(self.mean[z])
                         + '\nk0 = ' + str(self.k0[z]))
            info.pack()

        def openfile():
            filepath = filedialog.askopenfilename(initialdir='Desktop',
                                                  title='Select stack',
                                                  filetypes=(('tif files', '*.tiff *.tif'),
                                                             ('all files', '*.*')))
            if len(filepath) > 0:
                stack = io.imread(filepath)
                for z in range(0, len(stack)):
                    img = np.invert(img_as_ubyte(stack[z]))
                    try:
                        k = get_curvatures(img)
                        self.stack.append(img)
                        self.curves.append(k)
                        self.i.append(z)
                        k_max, k_min, k_mean = np.max(k), np.min(k), np.mean(k)
                        self.minmax.append([k_min, k_max])
                        self.mean.append(k_mean)
                    except IndexError:
                        pass
                self.stack = np.array(self.stack)
                self.curves = np.array(self.curves)
                self.colormap = get_scale(self.curves)
                self.minmax = np.array(self.minmax)
                self.k0 = np.zeros(len(self.i))
                textpath.config(text=os.path.split(filepath)[1])
                slice_z.delete(0, 'end')
                slice_z.insert(0, '0')
                plot()
                max_k_label.config(text='The maximum curvature is at z = ' +
                                        str(np.argmax(self.minmax.T[1])))
                min_k_label.config(text='The minimum curvature is at z = ' +
                                        str(np.argmin(self.minmax.T[0])))

        def plot():
            fig.clear()
            ax = fig.add_subplot(111)
            z = int(slice_z.get())
            img = self.stack[z]
            for j in range(0, len(img)):
                for i in range(0, len(img[0])):
                    if img[j, i] == 255:
                        img[j, i] = 150
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            ax.imshow(img, cmap='gray')
            x, y, u = get_splines(img)
            k, c = self.curves[z], self.colormap[z]
            # cmap = LinearSegmentedColormap.from_list('curves', sort_colors(c), len(c))
            cmap = get_subscale(c, k)
            ax.scatter(x, y, c=c, marker='.', linewidths=0)
            ax2 = fig.add_axes((0.85, 0.1, 0.02, 0.8))
            cb = ColorbarBase(ax2, cmap=cmap)
            ticklabels = [str(np.round(self.minmax[z][0], 3)),
                          str(np.round(np.median(k), 3)),
                          str(np.round(self.minmax[z][1], 3))]
            cb.set_ticks(np.linspace(0, 1, len(ticklabels)))
            cb.set_ticklabels(ticklabels)
            canvas.draw()
            toolbar.update()
            poly = Polygon(np.array([x, y]).T)
            self.k0[z] = np.sqrt(np.pi / poly.area)

        # atajos de teclado
        def key(event):
            if event.keysym == 'Left':
                left()
            if event.keysym == 'Right':
                right()
            if event.keysym == 'h':
                home()
            if event.keysym == 'Return':
                plot()

        open_button = Button(win, text='Select stack', command=openfile)
        open_button.pack()
        textpath = Label(win, text='- No stack has been selected -')
        textpath.pack()
        slices_frame = LabelFrame(win, text='Slices')
        slices_frame.pack()

        # altura en z:
        text_z = Label(slices_frame, text='z =')
        slice_z = Entry(slices_frame, width=4)
        go_to = Button(slices_frame, width=4, text='↵', command=plot)
        slice_z.bind('<Return>', key)
        text_z.grid(row=0, column=0)
        slice_z.grid(row=0, column=1)
        go_to.grid(row=0, column=2, padx=5)

        # cambiar z:
        def right():
            if len(self.curves) != 0:
                i = int(slice_z.get())
                slice_z.delete(0, 'end')
                if i != len(self.i) - 1:
                    slice_z.insert(0, str(i + 1))
                else:
                    slice_z.insert(0, str(0))
                plot()

        def left():
            if len(self.curves) != 0:
                i = int(slice_z.get())
                slice_z.delete(0, 'end')
                if i != 0:
                    slice_z.insert(0, str(i - 1))
                else:
                    slice_z.insert(0, str(len(self.i) - 1))
                plot()

        def home():
            if len(self.curves) != 0:
                slice_z.delete(0, 'end')
                slice_z.insert(0, '0')
                plot()

        right_but = Button(slices_frame, text='⮕', command=right, width=5)
        home_but = Button(slices_frame, text='⌂', command=home, width=5)
        left_but = Button(slices_frame, text='⬅', command=left, width=5)
        left_but.grid(row=0, column=3)
        home_but.grid(row=0, column=4)
        right_but.grid(row=0, column=5)
        win.bind('<Left>', key)
        win.bind('<Right>', key)
        win.bind('<h>', key)

        # info
        max_k_label = Label(slices_frame, text='The maximum curvature is at z =')
        min_k_label = Label(slices_frame, text='The minimum curvature is at z =')
        max_k_label.grid(row=1, columnspan=6)
        min_k_label.grid(row=2, columnspan=6)
        more_button = Button(slices_frame, text='More information', command=window_info, width=20)
        more_button.grid(row=3, columnspan=6)

        # figura
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win)
        canvas.get_tk_widget().pack()
        canvas.get_tk_widget().pack()

        win.mainloop()


Forces2D()
