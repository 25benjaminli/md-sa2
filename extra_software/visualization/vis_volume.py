import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import dotenv
import argparse
import nibabel as nib
import numpy as np
from skimage import io
from pathlib import Path
dotenv.load_dotenv(override=True)


parser = argparse.ArgumentParser()
parser.add_argument('--volume_num', type=int, default=74, help='which volume to visualize')
parser.add_argument('--vis_module', type=str, default='plt', help='plt or vedo to see the predictions')

args = parser.parse_args()
volume_num = args.volume_num  # so long as you have the file in the right position, you can view it. 
vis_module = args.vis_module

# ! recommended that you have already run the validation step of MD-SA2 or U-Net to generate the appropriate predictions
modality_list = ['t2f', 't1c', 't1n']
pred = np.load(f"volumes/aggregator/{str(volume_num).zfill(3)}_fold_0.npy")
label = np.load(f"{os.getenv('PREPROCESSED_PATH', '')}/BraTS-SSA-{str(volume_num).zfill(5)}-000/BraTS-SSA-{str(volume_num).zfill(5)}-000-seg.npy")
volumes = [nib.load(f"{os.getenv('DATASET_PATH', '')}/BraTS-SSA-{str(volume_num).zfill(5)}-000/BraTS-SSA-{str(volume_num).zfill(5)}-000-{modality}.nii.gz").get_fdata() for modality in modality_list]
print("label shape", label.shape) # (1, 224, 224, 155)
# convert label to (3, 224, 224, 155) one hot encoded by value 
label = np.concatenate([(label == i).astype(np.float32) for i in range(1, 4)], axis=0)
C, H, W, D = pred.shape 
print(f"Image shape: {pred.shape}") # (3, 224, 224, 155)
print(f"Label shape: {label.shape}") # (3, 224, 224, 155)

if vis_module == 'plt':
    print("Using matplotlib to visualize")
    # load the saved test image and label

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  
    plt.subplots_adjust(bottom=0.25)

    initial_slice = 0

    def update_plots(slice_idx):
        slice_idx = int(slice_idx)
        
        for i in range(3):
            for j in range(3):
                axes[i, j].clear()
        
        for channel in range(min(3, C)):
            axes[0, channel].imshow(volumes[channel][:, :, slice_idx], cmap='gray')
            axes[0, channel].set_title(f'Modality {modality_list[channel]} - Slice {slice_idx}')
            axes[0, channel].axis('off')

        for channel in range(min(3, C)):
            axes[1, channel].imshow(pred[channel, :, :, slice_idx], cmap='gray')
            axes[1, channel].set_title(f'Predicted Channel {channel} - Slice {slice_idx}')
            axes[1, channel].axis('off')
        
        for channel in range(min(3, C)):
            axes[2, channel].imshow(label[channel, :, :, slice_idx], cmap='gray')
            axes[2, channel].set_title(f'Label Channel {channel} - Slice {slice_idx}')
            axes[2, channel].axis('off')
        
        for channel in range(C, 3):
            axes[0, channel].axis('off')
            axes[1, channel].axis('off')
            axes[2, channel].axis('off')

        
        fig.canvas.draw_idle()

    update_plots(initial_slice)

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    depth_slider = Slider(ax_slider, 'Depth', 0, D-1, valinit=initial_slice, valstep=1)

    depth_slider.on_changed(update_plots)

    plt.tight_layout()
    plt.show()

elif vis_module == 'vedo':
    from vedo import Volume, show
    
    label_meshes = [Volume(label[channel]) for channel in range(label.shape[0])]
    pred_meshes = [Volume(pred[channel]) for channel in range(pred.shape[0])]
    modality_meshes = [Volume(volume) for volume in volumes]
    show(*label_meshes, *pred_meshes, *modality_meshes, N=9, axes=1, viewup="z")
    # add bar that lets you go one slice at a time? Maybe a slider?

elif vis_module == 'plotly':

    def read_pred(volume_orig):
        print("vol shape", volume_orig.shape)
        if volume_orig.shape[0] == 3:
            volume_orig = volume_orig.transpose(1, 2, 3, 0)
        elif volume_orig.shape[0] == 240:
            volume_orig = volume_orig.transpose(2, 0, 1, 3)
        volume_orig = volume_orig > 0.5

        r, c = volume_orig.shape[1], volume_orig.shape[2]

        volume_orig = volume_orig[..., ::-1]

        # Find the index of the maximum probability along the reversed channel dimension
        argmax_indices = np.argmax(volume_orig, axis=-1) + 1 # add 1 to each

        max_value_mask = np.max(volume_orig, axis=-1) == 1
        volume = np.zeros_like(argmax_indices, dtype=np.float32)
        volume[max_value_mask] = argmax_indices[max_value_mask] # this is the index of the max value

        max_val = volume.max()

        nb_frames = volume.shape[0]

        max_val = 3

        return volume, max_val, r, c, nb_frames, True

    def read_volume(volume_orig, modality=None):
        if modality is not None:
            volume_orig = volume_orig.squeeze() # (3, 141, 240, 240)
            print("volume orig shape", volume_orig.shape)
            volume_orig = volume_orig[modality] # (141, 240, 240)
            volume_orig = volume_orig[:, 8:-8, 8:-8] # cut
            
            r, c = volume_orig[0].shape
            max_val = volume_orig.max()
            nb_frames = volume_orig.shape[0]
        else:
            volume_orig = volume_orig.squeeze() # (3, 141, 240, 240)
            print("volume orig shape", volume_orig.shape)
            volume_orig = volume_orig[:, 8:-8, 8:-8] # cut
            
            r, c = volume_orig[0].shape
            max_val = volume_orig.max()
            nb_frames = volume_orig.shape[0]


        return volume_orig, max_val, r, c, nb_frames, False
    
    # ! every time you run, amke sure you update "mod_name"
    mod_name = "Modality t1c" # t1c, t1n, t2f
    # vis_slices_simultaneously = True # it's easier to explain if you just play around with it
    # volume, max_val, r, c, nb_frames, isPred = read_pred(pred) # either pred or volume. 
    volume, max_val, r, c, nb_frames, isPred = read_volume(volumes[0]) # either pred or volume. 

    # Define frames
    import plotly.graph_objects as go

    normed_ver = (volume - volume.min())/(volume.max() - volume.min())
    # print("normalized value of zeros", normed_ver.min(), normed_ver.max()) # check min and max

    # TODO: add second figure doing same thing but for other ones
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((nb_frames*0.1-0.1) - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1 - k]),
        cmin=0, cmax=max_val
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    def add_trace_to_fig(fig, slice_idx, scaler=0.1):
        """
        Sets the colorscale. The colorscale must be an array containing arrays mapping
        a normalized value to an rgb, rgba, hex, hsl, hsv, or named color string. 
        At minimum, a mapping for the lowest (0) and highest (1) values are required. 
        For example, [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]. To control the 
        bounds of the colorscale in color space, use cmin and cmax. 
        Alternatively, colorscale may be a palette name

        Determines whether or not the color domain is computed with respect to the input data 
        (here z or surfacecolor) or the bounds set in cmin and cmax Defaults to false when 
        cmin and cmax are set by the user.
        cmax Sets the upper bound of the color domain. 
        Value should have the same units as z or surfacecolor and if set, 
        cmin must be set as well.
    cmid

        Sets the opacityscale. The opacityscale must be an array
            containing arrays mapping a normalized value to an opacity
            value. At minimum, a mapping for the lowest (0) and highest (1)
            values are required. For example, `[[0, 1], [0.5, 0.2], [1,
            1]]` means that higher/lower values would have higher opacity
            values and those in the middle would be more transparent
            Alternatively, `opacityscale` may be a palette name string of
            the following list: 'min', 'max', 'extremes' and 'uniform'. The
            default is 'uniform'.
        """
        fig.add_trace(go.Surface(
        z=(nb_frames*0.1-slice_idx*scaler) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-slice_idx]),
        colorscale='Blackbody', # Blackbody for labels, Grayscale for imgs
        opacityscale=[[0, 0.05], [0.2, 0.9], [1, 1]],
        cmin=0, cmax=max_val, # 200
        colorbar=dict(thickness=20, ticklen=4)
        ))
    # TODO: change opacity of certain pixels

    # if vis_slices_simultaneously:
    #     for idx in range(5, 136, 5):
    #         if idx > 140:
    #             break
    #         add_trace_to_fig(fig, idx, scaler=0.1) # x2? make it seem diff
    #         # scaler puts the slices further apart
    # else:
    add_trace_to_fig(fig, 1, scaler=0.1) # x2? make it seem diff


    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]
    ax_style = dict(showbackground =True,
                    backgroundcolor="rgb(0,0,0)",

                    # showgrid=False,
                    zeroline=False,
                    gridcolor="rgb(255, 255, 255)"
                    )
    z_axstyle = dict(
        showbackground =True,
        backgroundcolor="rgb(0,0,0)",

        zeroline=False,
        range=[-0.1, nb_frames*0.1], 
        autorange=False,
        gridcolor="rgb(255, 255, 255)"

    )

    # print("vol num", vol_num, "mod idx", mod_name)
    # t1ce_120.nii

    fig.update_layout(
            title=f'Slices in volumetric data for volume {volume_num} for {mod_name}',
            width=600,
            height=600,
            scene=dict(
                xaxis=ax_style,
                yaxis=ax_style,
                zaxis=z_axstyle,
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 0, "t": 0}, # 10, 70
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders,
    )

    fig.show(renderer="browser")