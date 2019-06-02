from morphable_model import generate_face_images, U
from pinhole_camera_model import rotate_face, facial_landmarks
from optimization import test_optimization
from texturing import texturing
import matplotlib.pyplot as plt

# Task 2
"""
generate_face_images()
"""

# Task 3.1
# Result are saved in results/rotation
"""
w = [[0,10,0], [0,0,0], [0,-10,0]]
rotate_face(w)
"""

# Task 3.2
# Result is in results/facial_landmarks.png
"""
alpha = U(30)
delta = U(20)
w = [0,10,0]
t = [0,0,-400]
x = facial_landmarks(alpha, delta, w, t)
plt.scatter(x[0],x[1])
plt.savefig("./results/facial_landmarks.png")
"""

# Task 4
# Result in results/optimization
"""
test_optimization_one_image()
"""

# Task 5
# Result in results/texturing
"""
texturing()
"""

# Task 6

