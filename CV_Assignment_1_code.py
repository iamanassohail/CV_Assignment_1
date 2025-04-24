import cv2 as cv
import numpy as np
import random

def get_pixel_value(input_pixel, pos):
    """
    Returns the pixel value at a given position (sx, sy) in the image with bounds checking.
    """
    output_pixel = 0
    pos = np.round(pos)  # Round position to nearest integer
    if ((pos[0] < np.size(input_pixel, 0)) and (pos[0] >= 0) and
        (pos[1] < np.size(input_pixel, 1)) and (pos[1] >= 0)):
        output_pixel = input_pixel[int(pos[0]), int(pos[1])]  # Get pixel value if within bounds
    return output_pixel

def random_points(img_shape):
    """
    Generate 3 random non-colinear source and destination point pairs.
    """
    h, w = img_shape[:2]
    while True:
        src_pts = np.array([[random.randint(0, w-1), random.randint(0, h-1)] for _ in range(3)])
        if np.linalg.matrix_rank(np.hstack([src_pts, np.ones((3,1))])) == 3:
            break

    dst_pts = np.array([[random.randint(0, w-1), random.randint(0, h-1)] for _ in range(3)])
    return src_pts, dst_pts

def get_affine_matrix(src_pts, dst_pts):
    """
    Compute 2D affine transformation matrix using 3 points.
    """
    A = []
    B = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(xp)
        B.append(yp)
    
    A = np.array(A)
    B = np.array(B)

    # Solve for affine parameters
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Reshape into 2x3 affine matrix
    M = params.reshape(2, 3)
    return M

def apply_affine_transform(image, M, output_size):
    """
    Apply affine transformation to image.
    """
    h_out, w_out = output_size
    transformed = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    #Applying homogenous coordinates
    M_hom = np.vstack([M, [0, 0, 1]])  

    for y in range(h_out):
        for x in range(w_out):
            new_coord = np.matmul(M_hom,np.array([x, y, 1]))
            new_coord = new_coord / new_coord[2]
            sx, sy = new_coord[:2]
            transformed[y, x] = get_pixel_value(image, [sx, sy])
    
    return transformed

def annotate_points(image, points, color, offset=(0, 0)):
    """
    Annotate points with text showing their coordinates.
    """
    for (x, y) in points:
        text = f"({int(x)}, {int(y)})"
        cv.circle(image, (int(x) + offset[0], int(y) + offset[1]), 1, color, 3)
        # Draw the point and annotate it
        cv.putText(image, text, (int(x) + offset[0], int(y) + offset[1]), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv.LINE_AA)
        
if __name__ == '__main__':
    
    image = cv.imread('trail.jpg')  
    if image is None:
        print("Image not found.")
        exit()
    
    h, w = image.shape[:2]
    # Generate stable random source and destination points
    src_pts, dst_pts = random_points(image.shape)
    
    #manual src_pts, dst_pts 
    #dst_pts = (500,250),(802,1000),(100,650)
    #src_pts = (100,900),(500,50),(800,1000)
    #if np.linalg.matrix_rank(np.hstack([src_pts, np.ones((3,1))])) != 3:
        #print("Source points are collinear.")
        #exit()

    # Compute affine matrix
    M = get_affine_matrix(src_pts, dst_pts)
    
    # Apply transformation
    transformed_img = apply_affine_transform(image, M, (h, w))
    
    combined = np.hstack((image, transformed_img))
    
    # Annotate points
    annotate_points(combined, src_pts, (0, 0, 255))  # Red on original
    annotate_points(combined, dst_pts, (0, 255, 0), offset=(w, 0))  # Green on transformed

    #Results
    resized_img = cv.resize(combined, (1920, 1080), interpolation=cv.INTER_AREA)
    cv.imshow('Affine Transformation', resized_img)
    cv.imwrite('output_image.jpg', resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()