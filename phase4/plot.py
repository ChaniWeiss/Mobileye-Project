import matplotlib.pyplot as plt


def mark_tfl(image, candidates, fig, green_idx, title):
    fig.set_title(title)
    fig.imshow(image)
    if(green_idx > 0):
        fig.plot(candidates[:green_idx, [0]], candidates [:green_idx, [1]], 'ro', color='r', markersize=4)
    if(green_idx != len(candidates)):
        fig.plot(candidates[green_idx:, [0]], candidates[green_idx:, [1]], 'ro', color='g', markersize=4)


def mark_distances(image, candidates, fig, foe, distances, rot_pts):
    fig.set_title("distances")
    fig.imshow(image)
    fig.plot(candidates[:, 0],candidates[:, 1], 'b+')

    for i in range(len(candidates)):
        fig.plot([candidates[i, 0], foe[0]], [candidates[i, 1], foe[1]], 'b', linewidth=0.2)
        fig.text(candidates[i, 0], candidates[i, 1], r'{0:.1f}'.format(distances[i, 2]), color='r', fontsize=5)
    fig.plot(foe[0], foe[1], 'r+')
    fig.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
