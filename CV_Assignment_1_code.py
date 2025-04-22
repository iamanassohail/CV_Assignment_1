import cv2
import numpy as np
import random

def compute_affine_matrix(src_pts, dst_pts):
    # Compute a 3x3 affine transformation matrix from 3 source and 3 destination points.
    # src_pts and dst_pts are 3x2 numpy arrays.
    assert src_pts.shape == (3, 2)
    assert dst_pts.shape == (3, 2)

    A = []
    b = []

    for i in range(3):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append(u)
        b.append(v)

    A = np.array(A)
    b = np.array(b)

    # Solve for affine params
    affine_params = np.linalg.solve(A, b)

    # Build full 3x3 matrix in homogeneous coordinates
    affine_matrix = np.array([
        [affine_params[0], affine_params[1], affine_params[2]],
        [affine_params[3], affine_params[4], affine_params[5]],
        [0, 0, 1]
    ])

    return affine_matrix

def warp_image(img, affine_matrix, output_shape):
    # Apply an affine transformation to an image using numpy.
    # Only numpy allowed for transformation computation.
    h_out, w_out = output_shape
    result = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)

    inv_matrix = np.linalg.inv(affine_matrix)

    for y_out in range(h_out):
        for x_out in range(w_out):
            out_coord = np.array([x_out, y_out, 1])
            src_coord = inv_matrix @ out_coord
            x_src, y_src = src_coord[:2]

            if 0 <= x_src < img.shape[1] and 0 <= y_src < img.shape[0]:
                # Nearest neighbor sampling
                x_nn = min(max(int(round(x_src)), 0), img.shape[1] - 1)
                y_nn = min(max(int(round(y_src)), 0), img.shape[0] - 1)
                result[y_out, x_out] = img[y_nn, x_nn]


    return result

def draw_points(img, points, color, radius=5):
    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)

def run_example(image_path, src_pts, dst_pts, output_file='result.png'):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Compute affine transformation matrix
    affine_matrix = compute_affine_matrix(np.array(src_pts), np.array(dst_pts))

    # Warp the image using the affine transformation
    warped_img = warp_image(img, affine_matrix, (h, w))

    # Draw the source and destination points
    img_annotated = img.copy()
    warped_annotated = warped_img.copy()
    draw_points(img_annotated, src_pts, (0, 255, 0))  # Green
    draw_points(warped_annotated, dst_pts, (0, 0, 255))  # Red

    # Combine images side by side
    combined = np.hstack((img_annotated, warped_annotated))

    # Save result
    cv2.imwrite(output_file, combined)
    print(f"Result saved to {output_file}")

def random_test_runs(image_path, num_tests=5):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for i in range(num_tests):
        src_pts = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(3)]
        dst_pts = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(3)]
        run_example(image_path, src_pts, dst_pts, output_file=f'result_{i}.png')

# --- Example run ---
if __name__ == '__main__':
    # Load an image
    image_path = 'D:\Desktop\Sample_Image.png'  # Replace with your image path

    # Example source and destination points
    src_pts = [(50, 50), (200, 50), (50, 200)]
    dst_pts = [(70, 70), (220, 60), (80, 220)]

    # run_example(image_path, src_pts, dst_pts)
    # Optional: run random tests
    random_test_runs(image_path)
