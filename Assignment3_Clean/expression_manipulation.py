from PIL import Image, ImageSequence
from optimization import optimization_one_image
from mesh_to_png import triangles, mean_tex, mesh_to_png
from data_def import Mesh

from morphable_model import get_face_point_cloud

source_image = Image.open('./images/expression.gif')
target = Image.open('./images/first_frame.png')


target_model = optimization_one_image(300, # Can be changed to multiple images
                                target,
                                lambda_alpha=45,
                                lambda_delta=15,
                                lr=.128)
alpha = target_model.alpha

for i, frame in enumerate(ImageSequence.Iterator(source_image)):
    frame = frame.convert('RGB')
    model = optimization_one_image(300,
                                frame,
                                lr=.128)
    points = get_face_point_cloud(model.p, alpha, model.delta).view((-1, 3))
    mesh = Mesh(vertices=points.detach().numpy(), colors=mean_tex, triangles=triangles)
    mesh_to_png("./results/expression/frame_{}.png".format(i), mesh)
