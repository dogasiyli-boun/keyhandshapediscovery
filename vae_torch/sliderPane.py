import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

# def show_values(event, w_dict):
#     for _w in w_dict:
#         print(_w.get(), end=', ')
#     print(';')

class Slider_Class():
    def __init__(self, slider_count,
                 width=200, height=300,
                 title_str="LayoutGridClass"):
        # super(Slider_Class, self).__init__()

        self.slider_count = slider_count  # 3
        self.title_str = title_str
        self.width = width
        self.height = height
        self.img_id = 0

    def print_values(self):
        for k in self.w_dict:
            _w = self.w_dict[k]
            print(_w.get(), end='/')
        print(';')

        self.img_id = (self.img_id + 1) % 3
        if self.img_id == 1:
            img_path = '/home/wsubuntu/Downloads/DogaSiyli.png'
        elif self.img_id == 2:
            data = np.array(np.random.random((200, 200))*100, dtype=int)
            img_path = Image.frombytes('L', (data.shape[1], data.shape[0]), data.astype('b').tostring())
        else:
            img_path = '/home/wsubuntu/Downloads/luna_col.png'

        self.update_img_canvas(img_path)

    def create_slider(self):
        self.master_pane = tk.Tk()  # create root window
        self.master_pane.title(self.title_str)

        padx = 10
        pady = 5
        master_pane_width = 2*self.width + 2 * padx
        master_pane_height = 200 + self.slider_count*50 + 2 * pady
        self.master_pane.maxsize(master_pane_width, master_pane_height)
        self.master_pane.config(bg="skyblue")

        # Create left and right frames
        left_frame = tk.Frame(self.master_pane, width=self.width, height=self.height, bg='green')
        left_frame.grid(row=0, column=0, padx=20, pady=5)

        right_frame = tk.Frame(self.master_pane, width=self.width, height=self.height, bg='yellow')
        right_frame.grid(row=0, column=1, padx=5, pady=20)

        self.w_dict = {}
        for i in range(0, self.slider_count):
            self.w_dict["w" + str(i).zfill(2)] = tk.Scale(left_frame, from_=0, to=100, repeatinterval=100,
                                                          relief=tk.RAISED, orient=tk.HORIZONTAL, bg='purple')  # , command=show_values)
            self.w_dict["w" + str(i).zfill(2)].grid(row=i, column=0, padx=5, pady=5, sticky='w' + 'e' + 'n' + 's')  #

        _bt = tk.Button(left_frame, relief=tk.RAISED, text='Print Values',
                        command=self.print_values, bg='blue').grid(row=self.slider_count, column=0,
                                                              sticky='w' + 'e' + 'n' + 's')
        self.canvas = tk.Canvas(right_frame, width=200, height=200)
        self.canvas.place(x=-2, y=-2)
        img = Image.open('/home/wsubuntu/Downloads/luna_col.png').resize((200, 200))
        self.photo = ImageTk.PhotoImage(image=img)
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.grid(row=0, column=0)

        self.master_pane.update()
        self.master_pane.mainloop()
        return

    def update_img_canvas(self, img_path_or_data):
        if isinstance(img_path_or_data, str):
            img = Image.open(img_path_or_data).resize((200, 200))
        else:
            img = img_path_or_data
        self.photo = ImageTk.PhotoImage(image=img)
        # https://stackoverflow.com/questions/19838972/how-to-update-an-image-on-a-canvas
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
        self.master_pane.update()

# sc = Slider_Class(slider_count=6)
# sc.create_slider()
