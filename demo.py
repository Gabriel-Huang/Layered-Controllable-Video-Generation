from PIL import Image, ImageTk
from tkinter import *  
from PIL import ImageTk,Image  
from omegaconf import OmegaConf
import os
import tkinter as tk
from scripts.GUI_utils import sample_video, load_model_and_dset

class MainWindow():

    def __init__(self):
        self.master = tk.Tk()
        self.master.title('demo')
        # canvas for image
        self.canvas = tk.Canvas(self.master, width=512, height=256)
        self.canvas.pack(anchor=tk.NW)

        # load model
        ckpt = None
        ckpt = 'checkpoints/bair.ckpt'

        config_yaml = ['configs/bair.yaml']
        configs = [OmegaConf.load(cfg) for cfg in config_yaml]

        config = OmegaConf.merge(*configs)
        self.vqgan, global_step = load_model_and_dset(config, ckpt, gpu = True, eval_mode = True)

        # images

        self.current_frame_id = 0
        self.demo_image_folder = 'demo_img'

        # set first image on canvas

        self.x = 0
        self.y = 0
        self.move_speed = 30

        self.image = ImageTk.PhotoImage(Image.open(os.path.join(self.demo_image_folder, '00000.png'))) 
        # self.generate_frame()
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.image)

        # button to change image

        self.textBox=Text(self.master, height=2, width=10)
        self.textBox.pack(side=tk.RIGHT, anchor=tk.SE)
        buttonCommit=Button(self.master, height=1, width=10, text="Update Speed!", 
                            command=lambda: self.update_speed())
        buttonCommit.pack(side=tk.RIGHT, anchor=tk.SE)

        self.button_up = tk.Button(self.master, text="Up", command=self.on_click_up)
        self.button_up.pack()

        self.button_down = tk.Button(self.master, text="Down", command=self.on_click_down)
        self.button_down.pack()

        self.button_left = tk.Button(self.master, text="Left", command=self.on_click_left)
        self.button_left.pack()

        self.button_right = tk.Button(self.master, text="Right", command=self.on_click_right)
        self.button_right.pack()


        self.master.mainloop()

    def update_speed(self):
        inputValue=self.textBox.get("1.0",END)
        new_speed = int(inputValue)
        print(new_speed)
        self.move_speed = new_speed

    def on_click_up(self):
        self.y = self.move_speed
        self.x = 0
        self.generate_frame()
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)
        self.current_frame_id += 1

    def on_click_down(self):
        self.y = -self.move_speed
        self.x = 0
        self.generate_frame()
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)
        self.current_frame_id += 1

    def on_click_left(self):
        self.x = self.move_speed
        self.y = 0
        self.generate_frame()
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)
        self.current_frame_id += 1

    def on_click_right(self):
        self.x = -self.move_speed
        self.y = 0
        self.generate_frame()
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)
        self.current_frame_id += 1


    def generate_frame(self):
        sample_video(self.vqgan, self.demo_image_folder, self.current_frame_id, save_path = None, translate = (self.x, self.y))
        next_frame = Image.open(os.path.join(self.demo_image_folder, f'{(self.current_frame_id + 1):05}.png'))
        moved_mask = Image.open(os.path.join(self.demo_image_folder, f'{(self.current_frame_id + 1):05}_mask.png'))
        result = Image.new('RGB', (512, 256))
        result.paste(next_frame, (0, 0))
        result.paste(moved_mask, (next_frame.width, 0))
        self.image = ImageTk.PhotoImage(result) 

MainWindow()