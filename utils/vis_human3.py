import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                face = [int(vertex.split('/')[0]) - 1 for vertex in parts[1:]]
                faces.append(face)
    return vertices, faces

def plot_sampled_vertices(vertices, num_samples=5000, name='Sampled 3D Vertices'):
    if not isinstance(vertices, list) or not all(isinstance(v, (list, tuple)) and len(v) == 3 for v in vertices):
        raise ValueError("Input 'vertices' must be a list of [x, y, z] coordinates.")
    num_samples = min(num_samples, len(vertices))
    sampled_vertices = random.sample(vertices, num_samples)
    x = [v[0] for v in sampled_vertices]
    y = [v[1] for v in sampled_vertices]
    z = [v[2] for v in sampled_vertices]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='red', marker='o', s=1, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(name)
    
    # Set equal scaling
    max_range = max(
        np.ptp(x),  # Range of x
        np.ptp(y),  # Range of y
        np.ptp(z)   # Range of z
    )
    mid_x, mid_y, mid_z = np.mean(x), np.mean(y), np.mean(z)
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    # Set view to look down from +Z direction
    ax.view_init(elev=90, azim=-90)  # 90 degrees elevation, -90 azimuth for top-down XY view
    
    plt.show()

def load_and_plot(full_path, is_plot=True):
    if os.path.exists(full_path):
        try:
            vertices, faces = load_obj(full_path)
            print("Vertices loaded:", len(vertices))
            print("Faces loaded:", len(faces))
            
            # Split by '\\' and get the last part
            print(full_path)
            # file_name = full_path.split('\\')[-2]
            file_name = full_path.split('/')[-2]
            
            if (is_plot):
                plot_sampled_vertices(vertices, name=file_name)
        except Exception as e:
            print(f"Error while processing OBJ file: {e}")
    else:
        print(f"File not found: {full_path}")


def select_file():
    """Allows the user to select a file through a graphical file picker."""
    Tk().withdraw()  # Hides the root tkinter window
    file_path = askopenfilename(title="Select a 3D Object File", filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")])
    return file_path

def main():
    print("Please select a file to load.")
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    try:
        vertices, faces = load_obj(file_path)
        print(f"Loaded {len(vertices)} vertices and {len(faces)} faces.")
        plot_sampled_vertices(vertices)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()