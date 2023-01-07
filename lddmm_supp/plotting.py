import matplotlib.pyplot as plt



def show_warp_field(grid, interval=1, shape = (120, 120), size = (8, 8),limit_axis=True, show_axis=False, plot_separately = True):
    """
    utitlity function for plotting the warping grid 
    """
    if plot_separately:
        f, a = plt.subplots(1, 2, figsize = size)
        
        for x in range(0, shape[0], interval):
            a[0].set_axis_off()
            a[0].plot(grid[1, x, :], grid[0, x, :], 'k')
            #a[0].invert_yaxis()
            a[0].set_title("x field")
            a[0].set_aspect("equal")
        for y in range(0, shape[1], interval):
            a[1].set_axis_off()
            a[1].set_title("y field")
            a[1].plot(grid[1, :, y], grid[0, :, y], 'r')
            #a[1].invert_yaxis()
            a[1].set_aspect("equal")
        plt.gca().invert_yaxis()
        plt.show()
    else:
        plt.figure(figsize = size)
        if show_axis is False:
            plt.axis('off')
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for x in range(0, shape[0], interval):
            plt.plot(grid[1, x, :], grid[0, x, :], 'k')
        for y in range(0, shape[1], interval):
            plt.plot(grid[1, :, y], grid[0, :, y], 'k')
        #plt.gca().invert_yaxis()
        plt.show()

        
def plotx(fix, mov):

    f, a = plt.subplots(1, 2, figsize=(6, 3) )
    a[0].imshow(fix)
    a[1].imshow(mov)
    plt.show()