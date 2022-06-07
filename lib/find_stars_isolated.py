import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from utils import latex_settings


def find_coords_from_image(inf: int, sup: int, image: np.ndarray) -> np.ndarray:
    """Find all the pixel coordinates which have values in a given range.

    Args:
        inf (int): lower bound for image values;
        sup (int): upper bound for image values;
        image (np.ndarray): image choosen to be investigated.

    Returns:
        np.ndarray: coordinates of the wanted pixel values.
    """
    idys, idxs = np.where((image > inf) & (image < sup))
    ids = np.array((idxs, idys)).T
    return ids
    

try:
    from graph_tool.all import *
    graph_type = graph_tool.Graph
    def geometric_graph_stars(coords: np.ndarray, distance: float) -> graph_type:
        """Create a geometric graph starting from the (X,Y) coordinates,
        with a fixed distance as a threshold.

        Args:
            coords (np.ndarray): starts coordinates in the images;
            distance (float): radius threshold.

        Returns:
            graph_type: geometric graph for the stars coordinates.
        """
        # Create a geometric graph
        g, pos = geometric_graph(coords, distance)
        # Create a graph property with the vertices' position
        g.vp['pos'] = pos
        # Get the vertices' positions
        a = g.vp.pos.get_2d_array([0, 1]).T
        # Solve a bug of the code that invert y coordinates
        a[:,1] = -a[:,1]
        # Create a new vertex property with the correct positions
        new_pos = g.new_vp('vector<double>', vals=a)
        g.vp.new_pos = new_pos
        return g


    def save_graph_plot(Graph: graph_type, file: str) -> None:
        """Plot the Graph.

        Args:
            Graph (graph_type): graph containing all the information about
                                the vertices and edges as properties;
            file (str): file name of the output.
        """
        fig, ax = latex_settings()
        # Get the size in pixels for latex document plot
        size = fig.get_size_inches() * fig.dpi
        size = np.int32(size)
        graph_draw(g=Graph, pos=Graph.vp.new_pos, output_size=size, output=file)
        plt.close()
except:
    pass
    

@jit(nopython=True)
def find_IDs_from_data(dataset: np.ndarray, ids: np.ndarray,
                       npixel: float) -> np.ndarray:
    """Find all stars' ID in the dataset with the closest coordinates 
       respect the image indexis position. 

    Args:
        dataset (np.ndarray): 3 columns ndarray, with (Xcor, Y, ID)
                              dataset features;
        ids (np.ndarray): (X, y) image's coordinates;
        npixel (float): tolerance in pixels.

    Returns:
        List[float]: _description_
    """
    IDs = []
    n = len(ids)
    for i in range(n):
        mask = (dataset[:,0] - ids[i,0]) ** 2 + \
               (dataset[:,1] - ids[i,1]) ** 2 <= npixel ** 2
        if mask.sum():
            for ID in dataset[mask, -1]:
                IDs.append([ID, ids[i,0], ids[i,1]])
    return np.asarray(IDs)
