import argparse 
from psbody.mesh import Mesh 
from psbody.mesh import MeshViewer 
import cv2
import mediapipe as mp
import numpy as np
import math

def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument('mesh_filepath', type=str, help='The path to the mesh that you\'d like to get points for')
    parser.add_argument('texture_path', type=str, help='The path to the texture for the mesh')
    parser.add_argument('output_folder', type=str, help='The folder where outputs should be stored')

    return parser.parse_args()

def compute_rotation_matrix(angle: float):
    """
    Computes a rotation matrix around the y axis of the model given a radian angle. 
    This can be used to turn the mesh so that landmarks can be found for different views.
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

def compute_landmarks_for_image(image_path: str):
    """
    Given an image path, computes the landmarks for the face present in the frame
    """
    # This is adapted from https://google.github.io/mediapipe/solutions/face_mesh#python-solution-api
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=.5) as face_mesh:
        image = cv2.imread(image_path)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0] # Assume there is only one face in the mesh
        return None

if __name__ == '__main__':
    args = parse_args(argparse.ArgumentParser())

    # Start with your mesh
    my_mesh = Mesh(filename=args.mesh_filepath)
    
    # Add its texture
    my_mesh.set_texture_image(args.texture_path)
    
    # This step is required in order to get smooth shading
    my_mesh.reset_normals()

    viewer = my_mesh.show()

    for i in range(-20, 40, 1):
        my_mesh.rotate_vertices(compute_rotation_matrix(math.radians(i)))
        
        # Update the meshviewer after rotating the mesh
        my_mesh.show(mv=viewer)
        snapshot_name = f'{args.output_folder}/mesh_view_{i}.png'
        viewer.save_snapshot(snapshot_name)
        
        landmarks = compute_landmarks_for_image(snapshot_name)
        
        #mesh_raw_lndmks = [[lndmk.x, -lndmk.y, lndmk.z] for lndmk in landmarks.landmark]

        #my_mesh.set_landmarks_from_xyz(mesh_raw_lndmks)

        #my_mesh.show(mv=viewer)
        
        # Setup mediapipe drawing tools
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        annotated_image = cv2.imread(snapshot_name)
        
        mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
        cv2.imwrite(f'{args.output_folder}/media_pipe_view_{i}.png', annotated_image)

        # Rotate the mesh back for the next computation
        my_mesh.rotate_vertices(compute_rotation_matrix(math.radians(-i)))

    viewer.close()
