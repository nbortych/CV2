from morphable_model import generate_face_images, U
from pinhole_camera_model import facial_landmarks, rotate_face
import matplotlib.pyplot as plt

# Task 2
generate_face_images()

# Task 3.1
# Result are saved in results/rotation
w = [[0,10,0], [0,0,0], [0,-10,0]]
rotate_face(w)

# Task 3.2
# Results
alpha = U(30)
delta = U(20)
w = [0,10,0]
t = [0,0,-500]
x = facial_landmarks(alpha, delta, w, t)
plt.scatter(x[0],x[1])
plt.savefig("./results/facial_landmarks.png")
