from PIL import Image, ImageSequence
from optimization import optimization_one_image
from mesh_to_png import triangles, mean_tex, mesh_to_png
from data_def import Mesh

from morphable_model import get_face_point_cloud

source_image = Image.open('./images/expression.gif')
target = Image.open('./images/first_frame.png')

frames = []
for frame in ImageSequence.Iterator(source_image):
    frame = frame.convert('RGB')
    frames.append(frame)

alpha_model = optimization_one_image(300, # Can be changed to multiple images
                                frames[2],
                                lambda_alpha=45,
                                lambda_delta=15,
                                lr=.128)
alpha = alpha_model.alpha

deltas = []
Rs = []
ts = []
for frame in frames:
    model = optimization_one_image(50,
                                frame,
                                lambda_alpha=45,
                                lambda_delta=15,
                                lr=.128)
    deltas.append(model.delta)
    Rs.append(model.R)
    ts.append(model.t)
