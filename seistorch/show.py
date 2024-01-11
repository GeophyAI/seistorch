import numpy as np
from obspy import Trace, Stream
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SeisShow:

    def __init__(self,):
        pass

    def alternate(self, obs, syn, interval=10, trace_normalize=True, dt=0.001, dx=12.5, **kwargs):
        """Plot the observed and synthetic data in an alternating way.

        Args:
            obs (np.ndarray): The observed data.
            syn (np.ndarray): The synthetic data.
            interval (int, optional): The alternate interval . Defaults to 10.
            dt (float, optional): time step. Defaults to 0.001.
            dx (float, optional): trace step. Defaults to 12.5.
        """
        nt, nr = obs.shape
        show_data = np.zeros_like(obs)
        for i in range(0, nr, interval*2):
            range_obs = np.arange(i, min(i+interval, nr))
            range_syn = np.arange(i+interval, min(i+2*interval, nr))
            show_data[:,range_obs] = obs[:,range_obs]
            show_data[:,range_syn] = syn[:,range_syn]

        if trace_normalize:
            show_data /= np.max(np.abs(show_data), axis=0, keepdims=True)

        fig, ax = plt.subplots(1,1,figsize=(8,4))
        vmin, vmax=np.percentile(show_data, [2,98])
        ax.imshow(show_data, 
                  vmin=vmin, 
                  vmax=vmax, 
                  extent=(0, nr*dx, nt*dt, 0), 
                  aspect="auto",
                  **kwargs)
                # ax.text(0.00, 0.95, "obs", 
                # transform=ax.transAxes, 
                # color='w', fontsize=14, 
                # fontweight='bold')
        # show text on the image
        kwargs_text = dict(color='w', fontsize=14, fontweight='bold')
        for i in range(0, nr, interval*2):
            ax.text(i*dx, 0.25, "obs", **kwargs_text)
            ax.text((i+interval)*dx, 0.25, "syn", **kwargs_text)
        
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def arrival(self, data, arrival, dt=0.001, dh=12.5, figsize=(5,6)):
        """Plot the data with the arrival time.

        Args:
            data (np.ndarray): The data to be plotted.
            arrival (np.ndarray): The arrival time.
        """
        fig, ax = plt.subplots(1,1,figsize=figsize)
        nt, nr, nc = data.shape
        assert arrival.size==nr, "The arrival time should have the same size as the number of traces."
        vmin, vmax = np.percentile(data, [2,98])
        kwargs = dict(vmin=vmin, 
                      vmax=vmax, 
                      extent=(0, nr*dh, nt*dt, 0), 
                      cmap='seismic',
                      aspect="auto")
        ax.imshow(data, **kwargs)
        x = np.arange(nr)*dh
        ax.scatter(x, arrival*dt, color='black', s=2)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def geometry(self, vel, sources:list, receivers:list, savepath:str, dh=1, interval=1):
        """Plot the velocity model and the source and receiver list.

        Args:
            vel (np.ndarray): The velocity model.
            sources (list): The source list.
            receivers (list): The receiver list.
            savepath (str): The path to save the gif figure.
            dh (int, optional): The grid step in both x and z. Defaults to 20.
            interval (int, optional): The interval between frames. Defaults to 1.

        """
        dim_of_vel = vel.ndim
        dim_of_src = len(sources[0])
        dim_of_rec = len(receivers[0])

        assert dim_of_vel==2, f"The velocity model should be 2D, but got {dim_of_vel}D"

        nz, nx = vel.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(vel, cmap='seismic', aspect="auto", extent=[0, nx*dh, nz*dh, 0])
        sc_sources = ax.scatter([], [], c='red', marker="v", s=10, label='Sources')
        sc_receivers = ax.scatter([], [], c='blue', marker="^", s=2, label='Receivers')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Sources: {len(sources)}, Receivers: {len(receivers[0][0])}")
        plt.tight_layout()

        # define the figure
        def update(frame):
            
            sc_sources.set_offsets(np.stack(sources[frame][:2], axis=0).T*dh)
            sc_receivers.set_offsets(np.stack(receivers[frame][:2], axis=1)*dh)

            return sc_sources, sc_receivers

        ani = FuncAnimation(fig, update, frames=len(sources), interval=interval)
        ani.save(savepath, writer='imagemagick')  

    def np2st(self, data, dt=0.001, dx=12.5, downsample=4):
        """Convert numpy array to obspy stream.

        Args:
            data (np.ndarray): The data to be converted.
            dt (time interval, optional): Time interval. Defaults to 0.001.

        Returns:
            obspy.Stream: The converted stream.
        """
        nt, nr, _ = data.shape
        traces = []
        for i in range(0, nr, downsample):
            traces.append(Trace(data=data[:,i,0], header={"delta":dt, "distance": i*dx}))
        return Stream(traces=traces)

    def section(self, data, dx=12.5, dz=12.5, savepath=None, **kwargs):
        """Plot a section of the data.

        Args:
            data (np.ndarray): The data to be plotted.
            dx (float, optional): Spatial interval in x direction. Defaults to 12.5.
            dz (float, optional): Spatial interval in z direction. Defaults to 12.5.
        """
        fig, ax = plt.subplots(1,1,figsize=(10,6))
        vmin, vmax=np.percentile(data, [2,98])
        ax.imshow(data, vmin=vmin, vmax=vmax, extent=(0, data.shape[1]*dx, data.shape[0]*dz, 0), **kwargs)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        plt.tight_layout()
        plt.show()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches='tight')

    def shotgather(self,
                   datalist: list, 
                   titlelist: list=[], 
                   figsize=(10,6),
                   inacolumn=False,
                   inarow=False,
                   normalize=True,
                   savepath="shotgather.png",
                   dx=12.5,
                   dt=0.001,
                   colorbar=True,
                   show=False,
                   **kwargs):
        
        if inacolumn: ncols, nrows = (1, len(datalist))
        elif inarow: ncols, nrows = (len(datalist), 1)

        assert any([inacolumn, inarow]), "Please specify the plot layout."

        fig, axes = plt.subplots(nrows,ncols,figsize=figsize)
        
        if datalist[0].ndim==2:
            nt, nr = datalist[0].shape
        elif datalist[0].ndim==3:
            nt, nr, _ = datalist[0].shape

        extent = (0, nr*dx, nt*dt, 0)
        if normalize:            
            vmin, vmax=np.percentile(datalist[0], [2,98])
            
        axes = axes.ravel() if ncols*nrows>1 else [axes]

        for d, ax, title in zip(datalist, axes, titlelist):
            if d.ndim==3 and d.shape[2]==1: d = d[:,:,0]
            if not normalize: vmin, vmax=np.percentile(d, [2,98])
            ax.imshow(d, vmin=vmin, vmax=vmax, extent=extent, aspect="auto", **kwargs)
            ax.set_title(title)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("t (s)")
            if colorbar: plt.colorbar(ax.images[0], ax=ax)

        plt.tight_layout()
        if show:
            plt.show()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches='tight')

    def spectrum(self, datalist: list, labellist: list, dt=0.001, db=False, endfreq=100, normalize=False):
        """Compute the frequency spectrum of the data.

        Args:
            datalist (list): The data to be computed, the nt should be the first dimension.
            dt (float, optional): Time interval. Defaults to 0.001.
            db (bool, optional): Whether to plot in dB scale. Defaults to False.
            endfreq (int, optional): The end frequency to be plotted. Defaults to 100.
        Returns:
            np.ndarray: The frequency spectrum.
            np.ndarray: The frequency axis.
        """
        nt = datalist[0].shape[0]
        for idx, d in enumerate(datalist):
            if d.ndim==1: d = d.reshape(-1,1)
            fft_freq = np.fft.fft(d, axis=0)
            freqs = np.fft.fftfreq(nt, dt)
            freq_interval = 1/(dt*nt)
            end_freq_index = int(endfreq//freq_interval)
            size = freqs.size//2
            # Sum traces
            amp = np.sum(np.abs(fft_freq), axis=1)

            if db:
                amp = 20*np.log10(amp)

            if normalize:
                amp /= amp.max()

            label = labellist[idx] if labellist else ""
            plt.plot(freqs[:end_freq_index], 
                     amp[:end_freq_index], label=label)
        
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    def wiggle(self, 
               datalist: list, 
               colorlist: list, 
               labellist: list=[],
               dt=0.001, dx=12.5, downsample=4, 
               fontsize=14,
               savepath=None, 
               show=False,
               **kwargs):
        """Wiggle plot for the data.

        Args:
            datalist (list): A list of data to be plotted.
            colorlist (list): A list of colors.
            labellist (list, optional): A list of labels. Defaults to [].
            dt (float, optional): Time interval. Defaults to 0.001.
            dx (float, optional): Spatial interval. Defaults to 12.5.
            savepath ([type], optional): The path to save the figure. Defaults to None.
        """
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        kwargs=dict(type='section',
                    fig=fig, 
                    ax=ax, 
                    alpha=1.,
                    linewidth=1.5, 
                    time_down=True)
        # Plot wiggles
        for d, c in zip(datalist, colorlist):
            st = self.np2st(d, dt, dx, downsample)
            st.plot(color=c, **kwargs)
        # Set labels
        lines = ax.get_lines()
        ntraces = len(lines)//len(datalist)
        for idx, line in enumerate(lines):
            if idx%ntraces==0 and labellist:
                line.set_label(labellist[idx//ntraces])
        # set fontsize of tick labels
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # set fontsize of x and y labels
        ax.set_xlabel("Offset (km)", fontsize=fontsize)
        ax.set_ylabel("Time (s)", fontsize=fontsize)
        # Set legend
        plt.legend(loc='upper right', fontsize=fontsize)
        plt.tight_layout()
        if show:
            plt.show()
        # Save figure
        if savepath:
            fig.savefig(savepath, dpi=300)
            plt.close()

