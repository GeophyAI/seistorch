import numpy as np
from obspy import Trace, Stream
import matplotlib.pyplot as plt

class SeisShow:

    def __init__(self,):
        pass

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

    def wiggle(self, 
               datalist: list, 
               colorlist: list, 
               labellist: list=[],
               dt=0.001, dx=12.5, downsample=4, 
               savepath=None, 
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
        # Set legend
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        # Save figure
        if savepath:
            plt.savefig(savepath, dpi=300)
            plt.close()

    def shotgather(self,
                   datalist: list, 
                   titlelist: list=[], 
                   figsize=(10,6),
                   inacolumn=False,
                   inarow=False,
                   dx=12.5,
                   dt=0.001,
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
        for d, ax, title in zip(datalist, axes.ravel(), titlelist):
            if d.ndim==3 and d.shape[2]==1: d = d[:,:,0]
            vmin, vmax=np.percentile(d, [2,98])
            ax.imshow(d, vmin=vmin, vmax=vmax, extent=extent, **kwargs)
            ax.set_title(title)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("t (s)")

        plt.tight_layout()
        plt.show()

    def section(self, data, dx=12.5, dz=12.5, **kwargs):
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


