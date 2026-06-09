import napari, numpy as np
v = napari.Viewer()
v.add_image(np.zeros((5,2,64,64), dtype=np.float32), channel_axis=1, colormap=['green','red'])
napari.run()