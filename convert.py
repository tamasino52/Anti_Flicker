from moviepy.editor import *

VideoFileClip('./output/input.MP4').speedx(1).write_gif('output.gif')