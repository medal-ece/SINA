import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def apply_linelabels(image,annotation_labels):

    height, width = image.shape[:2]

    # Create a matplotlib figure with the same size as the image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    ax.imshow(image, cmap='gray', aspect='equal')  # Preserve aspect ratio
    ax.axis('off')  # Hide axes

    # Annotate the image with cluster labels at the centroids
    for mean_col,mean_row,label in annotation_labels:
        ax.text(mean_col, mean_row, f'{label + 1}', color='red', fontsize=12, ha='center')

    # Save the plot to a buffer without changing the image dimensions
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Use PIL to open the buffer and convert it to a numpy array
    pil_image = Image.open(buf)
    annotated_img = np.array(pil_image)

    # Close the figure to release memory
    plt.close(fig)

    return annotated_img

def radial_grouping_new(image, line_segments, image_size):
    """
    Groups connected line segments and annotates the image with cluster labels at each line segment.
    The annotated image is saved as 'grouped_img.png'.

    Parameters:
    - image: NumPy ndarray representing the image.
    - line_segments: List of tuples, where each tuple contains two endpoints (arrays) of a line segment.
                     Each endpoint should be in (row, col) format.
    - image_size: Tuple (height, width) specifying the desired size to resize the image.

    Returns:
    - annotated_image: Image with annotated line segments.
    - annotation_labels: List of dictionaries with line segment midpoints and their labels.
    """

    def is_within_radius(point1, point2, radius=40):
        """Check if two points are within the given radius."""
        return np.linalg.norm(np.array(point1) - np.array(point2)) <= radius

    # List to store clusters
    clusters = []
    visited = set()  # To keep track of visited line segments

    # Group line segments using nested loops
    for i in range(len(line_segments)):
        if i in visited:
            continue
        
        # Start a new cluster
        cluster = [line_segments[i]]
        visited.add(i)

        # Check other line segments
        for j in range(i + 1, len(line_segments)):
            if j in visited:
                continue

            # Get points of current line segments
            for line1 in cluster:
                (p1_start, p1_end) = line1
                (p2_start, p2_end) = line_segments[j]

                # Check if any of the endpoints are within the radius
                if (is_within_radius(p1_start, p2_start) or
                    is_within_radius(p1_start, p2_end) or
                    is_within_radius(p1_end, p2_start) or
                    is_within_radius(p1_end, p2_end)):
                    # Add the line segment to the current cluster
                    cluster.append(line_segments[j])
                    visited.add(j)
                    break  # No need to check further points for this segment

        # Add the current cluster to the list of clusters
        clusters.append(cluster)

    # Resize the image to the specified size
    image = cv2.resize(image, (image_size[1], image_size[0]))
    height, width = image.shape[:2]

    # Create a matplotlib figure with the same size as the image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    ax.imshow(image, cmap='gray', aspect='equal')  # Preserve aspect ratio
    ax.axis('off')  # Hide axes

    annotation_labels = []
    # Annotate the image with cluster labels at each line segment
    for label, cluster in enumerate(clusters):
        for line_segment in cluster:
            (p1, p2) = line_segment
            # Calculate the midpoint of the line segment
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            # Annotate the image with the cluster label
            ax.text(midpoint[1], midpoint[0], f'{label + 1}', color='red', fontsize=12, ha='center')
            # Collect annotation information
            annotation_labels.append([p1[1], p1[0], label + 1])

    # Save the plot to a buffer without changing the image dimensions
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Use PIL to open the buffer and convert it to a numpy array
    pil_image = Image.open(buf)
    annotated_image = np.array(pil_image)

    # Close the figure to release memory
    plt.close(fig)

    return annotated_image, annotation_labels


def radial_grouping(image, line_segments, image_size):
    """
    Groups connected line segments and annotates the image with cluster labels at cluster centroids.
    The annotated image is saved as 'grouped_img.png'.

    Parameters:
    - image: NumPy ndarray representing the image.
    - line_segments: List of tuples, where each tuple contains two endpoints (arrays) of a line segment.
                     Each endpoint should be in (row, col) format.
    - image_size: Tuple (height, width) specifying the desired size to resize the image.

    Returns:
    - clusters: Dictionary where keys are cluster labels and values are cluster centroids (row, col).
    """

    def is_within_radius(point1, point2, radius=30):
        """Check if two points are within the given radius."""
        return np.linalg.norm(np.array(point1) - np.array(point2)) <= radius

    # List to store clusters
    clusters = []
    visited = set()  # To keep track of visited line segments

    # Group line segments using nested loops
    for i in range(len(line_segments)):
        if i in visited:
            continue
        
        # Start a new cluster
        cluster = [line_segments[i]]
        visited.add(i)

        # Check other line segments
        for j in range(i + 1, len(line_segments)):
            if j in visited:
                continue

            # Get points of current line segments
            for line1 in cluster:
                (p1_start, p1_end) = line1
                (p2_start, p2_end) = line_segments[j]

                # Check if any of the endpoints are within the radius
                if (is_within_radius(p1_start, p2_start) or
                    is_within_radius(p1_start, p2_end) or
                    is_within_radius(p1_end, p2_start) or
                    is_within_radius(p1_end, p2_end)):
                    # Add the line segment to the current cluster
                    cluster.append(line_segments[j])
                    visited.add(j)
                    break  # No need to check further points for this segment

        # Add the current cluster to the list of clusters
        clusters.append(cluster)

    # Calculate centroids for each cluster
    cluster_centroids = {}
    for idx, cluster in enumerate(clusters):
        points = []
        for line_segment in cluster:
            points.extend(line_segment)  # Add both endpoints of the line segment to points
        points = np.array(points)
        centroid = points.mean(axis=0)  # Calculate the mean point (centroid)
        cluster_centroids[idx] = (centroid[0], centroid[1])

    # Resize the image to the specified size
    image = cv2.resize(image, (image_size[1], image_size[0]))
    height, width = image.shape[:2]

    # Create a matplotlib figure with the same size as the image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    ax.imshow(image, cmap='gray', aspect='equal')  # Preserve aspect ratio
    ax.axis('off')  # Hide axes

    annotation_labels = []
    # Annotate the image with cluster labels at the centroids
    for label, mean_point in cluster_centroids.items():
        mean_row, mean_col = int(mean_point[0]), int(mean_point[1])  # Centroid coordinates
        # Annotate the image with the cluster label
        ax.text(mean_col, mean_row, f'{label + 1}', color='red', fontsize=12, ha='center')
        annotation_labels.append([mean_col,mean_row,label+1])

    # Save the plot to a buffer without changing the image dimensions
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Use PIL to open the buffer and convert it to a numpy array
    pil_image = Image.open(buf)
    annotated_image = np.array(pil_image)

    # Close the figure to release memory
    plt.close(fig)

    return annotated_image,annotation_labels

def cluster_grouping(image, line_segments, image_size):
    """
    Groups connected line segments and annotates the image with cluster labels at cluster centroids.
    The annotated image is saved as 'grouped_img.png'.

    Parameters:
    - image: NumPy ndarray representing the image.
    - line_segments: List of tuples, where each tuple contains two endpoints (arrays) of a line segment.
                     Each endpoint should be in (row, col) format.
    - image_size: Tuple (height, width) specifying the desired size to resize the image.

    Returns:
    - clusters: Dictionary where keys are cluster labels and values are cluster centroids (row, col).
    """
    # Resize the image to the specified size
    image = cv2.resize(image, (image_size[1], image_size[0]))
    height, width = image.shape[:2]

    N = len(line_segments)
    adj_list = {i: set() for i in range(N)}
    epsilon = 1e-1

    def is_point_on_line_segment(P, Q1, Q2, epsilon=1e-6):
        # P, Q1, Q2 are in (row, col) format
        # Convert to (col, row) for computation
        P_x, P_y = P[1], P[0]
        Q1_x, Q1_y = Q1[1], Q1[0]
        Q2_x, Q2_y = Q2[1], Q2[0]

        # Compute cross product
        s = (P_y - Q1_y) * (Q2_x - Q1_x) - (P_x - Q1_x) * (Q2_y - Q1_y)
        if abs(s) > epsilon:
            return False

        # Compute dot product
        dot = (P_x - Q1_x) * (Q2_x - Q1_x) + (P_y - Q1_y) * (Q2_y - Q1_y)
        len_sq = (Q2_x - Q1_x) ** 2 + (Q2_y - Q1_y) ** 2
        if dot < 0 or dot > len_sq:
            return False

        return True

    # Check for connections between line segments
    for i in range(N):
        P1_i, P2_i = line_segments[i]
        for j in range(i + 1, N):
            P1_j, P2_j = line_segments[j]
            connected = False
            # Check endpoints of line i on line j
            for P_i in [P1_i, P2_i]:
                if is_point_on_line_segment(P_i, P1_j, P2_j, epsilon):
                    connected = True
                    break
            if not connected:
                # Check endpoints of line j on line i
                for P_j in [P1_j, P2_j]:
                    if is_point_on_line_segment(P_j, P1_i, P2_i, epsilon):
                        connected = True
                        break
            if connected:
                adj_list[i].add(j)
                adj_list[j].add(i)

    # Find connected components using DFS
    visited = set()
    components = []

    def dfs(node, adj_list, visited, component):
        visited.add(node)
        component.add(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                dfs(neighbor, adj_list, visited, component)

    for i in range(N):
        if i not in visited:
            component = set()
            dfs(i, adj_list, visited, component)
            components.append(component)

    # Calculate cluster centroids (mean of midpoints)
    clusters = {}
    for idx, component in enumerate(components):
        cluster_points = []
        for line_idx in component:
            P1, P2 = line_segments[line_idx]
            midpoint = (P1 + P2) / 2  # Midpoint in (row, col)
            cluster_points.append(midpoint)
        cluster_points = np.array(cluster_points)
        cluster_mean = np.mean(cluster_points, axis=0)
        clusters[idx] = cluster_mean  # Cluster centroid in (row, col)

    # Create a matplotlib figure with the same size as the image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    ax.imshow(image, cmap='gray', aspect='equal')  # Preserve aspect ratio
    ax.axis('off')  # Hide axes

    # Annotate the image with cluster labels at the centroids
    for label, mean_point in clusters.items():
        mean_row, mean_col = int(mean_point[0]), int(mean_point[1])  # Centroid coordinates
        # Annotate the image with the cluster label
        ax.text(mean_col, mean_row, f'{label + 1}', color='red', fontsize=12, ha='center')

    # Remove margins and prevent automatic resizing
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure as 'grouped_img.png' without changing the image dimensions
    plt.savefig('grouped_img.png', format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Return the clusters dictionary
    return clusters