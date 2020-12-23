from phase3.SFM_standAlone import FrameContainer
import numpy as np
from phase3 import SFM
import phase4.plot as plot


def visualize(focal, pp, prev_container, curr_container, fig):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))
    plot.mark_distances(curr_container.img, curr_container.traffic_light, fig, foe, curr_container.traffic_lights_3d_location, rot_pts)


def calc_distances(img_path, data_holder, curr_candidates, prev_candidates, fig):
    curr_container, prev_container = FrameContainer(img_path), FrameContainer()
    curr_container.traffic_light, prev_container.traffic_light = np.array(curr_candidates), np.array(prev_candidates)
    curr_container.EM = data_holder.EM
    curr_container = SFM.calc_TFL_dist(prev_container, curr_container, data_holder.focal, data_holder.pp)
    visualize(data_holder.focal, data_holder.pp, prev_container, curr_container, fig)
    return curr_container.traffic_light


