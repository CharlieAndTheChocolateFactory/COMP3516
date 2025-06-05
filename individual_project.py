from scipy.signal import butter, sosfiltfilt
from scipy.signal import savgol_filter
from hampel import hampel
import numpy as np
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class CPD():
    def __init__(self, data_file_p=None):
        self.data_file_p = data_file_p
        
        if self.data_file_p is not None:
            self.data = self._load_data(self.data_file_p)

    def _load_data(self, data_file_p):
        with open(data_file_p, 'rb') as f:
            self.data = pickle.load(f)
        self.csi = self.data['CSI'] # CSI data, shape (N_f, N_t)
        self.fs = self.data['fs'] # sampling rate
        self.frame_len = self.data['frame_len'] # frame length of one CSI data
        return self.data
    
    
    def get_csi_rate(self):
        return self.csi_rate
    
    
    def get_ACF(self, wl=7, ws=0.2):
        """
        Calculates the Autocorrelation Function (ACF) matrix for Channel State Information (CSI) data.

        This method computes the ACF matrix for CSI data using specified window length and step. 
        It also determines the CSI sampling rate, which is crucial for accurate ACF calculation.

        Parameters:
        - wl: float, optional (default is 7)
            The window length in seconds for computing the ACF. Specifies how long each window of data 
            should be for the ACF calculation.
        - ws: float, optional (default is 0.2)
            The window step in seconds. Determines the step size or how much the window moves 
            for each ACF calculation.

        Returns:
        - self.acf: array_like (len(self.lag), len(self.t_s), N_f)
            The calculated ACF matrix stored within the class instance. Each row in the matrix 
            corresponds to the ACF of a window of CSI data.
        - self.t_s: array_like (in seconds)
            The time points corresponding to the ACF matrix. This is a vector of time points 
            at which the ACF matrix is calculated.
        - self.lag: array_like (in seconds)
            The lag points corresponding to the ACF matrix. This is a vector of lag points 
            at which the ACF matrix is calculated.

        Note:
        The CSI sampling rate (`self.csi_rate`) is different from the acoustic sampling rate and is 
        calculated internally to determine the number of CSI frames per second. This rate is crucial for 
        understanding the temporal resolution of the CSI data and correctly computing the ACF matrix.
        """
        self.acf = None # store your ACF matrix
        self.csi_rate = None # store CSI sampling rates
        self.t_s = None # store time points
        self.lag = None # store lag points
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
                # calculate CSI sampling rates
        from scipy.signal import correlate
        self.csi_rate = self.fs / self.frame_len #CSI sample per second

        # Determine number of samples per ACF window and step size in samples
        wl_samples = int(wl * self.csi_rate)
        ws_samples = int(ws * self.csi_rate)

        # Initialize time points for ACF computation
        self.t_s = np.arange(0, self.csi.shape[1]- wl_samples , ws_samples) / self.csi_rate
        # print("self.t_s:" , self.t_s.shape)

        self.lag = np.arange(0, wl_samples) / self.csi_rate # lag points for ACF 
        # print("self.lag:" , self.lag.shape)

        # print("self.lag:" , len(self.lag))
        # why 329?

        # Define the maximum lag as half the window length (typical choice for ACF)
        # max_lag = wl_samples // 2

        # Initialize ACF matrix
        self.acf = np.zeros((len(self.lag), len(self.t_s), self.csi.shape[0]))

        # Autocorrelation function 
        def autocorrelation(x):
            result = correlate(x, x, mode='full')
            result /= np.max(result)
            result = result[result.size // 2:]
            return result
        

        # # Compute ACF for each subcarrier
        # for sub_carrier in range(self.csi.shape[0]):
        #     for t in enumerate(len(self.t_s - wl_samples)):
        #         for lag in range(len(self.lag)):
        #             window = self.csi[sub_carrier, t:t+wl_samples]
        #             self.acf[lag, t, sub_carrier] = autocorrelation(window)


        # Compute ACF for each subcarrier
        for sub_carrier in range(self.csi.shape[0]):
            for i, t in enumerate(range(0, self.csi.shape[1] - wl_samples, ws_samples)):
                # Extract CSI data for current window
                window = self.csi[sub_carrier, t:t + wl_samples]
                window = window - np.mean(window)

                # Compute ACF for the window
                acf = autocorrelation(window)

                # Normalize ACF and store in matrix
                self.acf[:, i, sub_carrier] = acf[:len(self.lag)]
        
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        return self.acf, self.t_s, self.lag
    
    def get_ms(self):
        """
        Calculates the motion statistic (ms) and its average across subcarriers (ms_bar).

        This method computes the motion statistic `g(f, t)` for each subcarrier using the autocorrelation 
        function (ACF) at a lag of one over the CSI sampling rate (`Fs`). It then averages these 
        statistics over all subcarriers to obtain `\overline{g(f, t)}`.

        The motion statistic `g(f, t)` is a measure of temporal variation in the CSI data, which can be 
        indicative of motion within the environment.


        Returns:
        - self.ms: array_like (len(self.lag), N_f)
            The motion statistic `g(f, t)` for each subcarrier. This is a vector where each element 
            corresponds to the motion statistic of a subcarrier.
        
        - self.ms_bar: array_like (len(self.t_s), )
            The average motion statistic `\overline{g(f, t)}` across all subcarriers. This is a single 
            scalar value representing the mean motion statistic.

        """
        self.ms = None # store ms matrix
        self.ms_bar = None # store average ms
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<

        # calculate the motion statistics (ms)
        # Initialize motion statistic matrix
        self.ms = np.zeros((len(self.t_s), self.csi.shape[0]))
        # Compute motion statistic for each subcarrier
        self.ms = self.acf[1, :, :]

        # Calculate average motion statistic
        self.ms_bar = np.mean(self.ms, axis=1)
        
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        return self.ms, self.ms_bar
    
    
    
    def get_br(self):
        """
        Determines the breathing rate (br) from the autocorrelation function (ACF) matrix of 
        Channel State Information (CSI) data by dynamically combining subcarriers to maximize signal SNR.

        This function combines various subcarriers within the ACF matrix, focusing on enhancing the 
        signal's first prominent peak to reliably detect the breathing rate. Different weighting strategies 
        for combining subcarriers are encouraged to optimize the detection of the first peak, indicating 
        the breathing rate.

        Returns:
        - combined_acf: array_like (len(self.lag), len(self.t_s))
            The combined ACF after subcarrier optimization, aimed at enhancing the signal-to-noise ratio (SNR).
        
        - peak_index: array_like (len(self.t_s), )
            The indices (index) of the detected prominent peaks within the combined ACF, focusing on the first 
            significant peak corresponding to the breathing rate.
        
        - br_t: array_like (len(self.t_s), )
            The detected breathing rates over time, derived from the position of the first prominent peak 
            in the combined ACF.
        
        - average_br: float     
            The average breathing rate calculated from the detected breathing rates over time.

        Notes:
        - Subcarrier combination aims to enhance the ACF matrix by emphasizing the first peak, which is 
        critical for accurate breathing rate detection. Not all subcarriers need to be combined; a 
        selective approach (e.g., `topK` subcarriers) may yield better results.
        
        - The breathing rate range considered is 12-60 BPM, reflecting typical values for children and adults.
        
        - Different methods for peak detection, including `find_peaks()` or custom algorithms, can be used. 
        Care should be taken with parameter selection to ensure accurate peak identification.
        """
        combined_acf = None
        peak_index = []
        br_t = []
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        from scipy.signal import find_peaks
        topK = 7
        # find the mean of motion statistic of each subcarrier and reduce it to 1D array by averaging self.t_s
        ms_mean = np.mean(self.ms, axis=0) #axis=0 means averaging the time

        # sort the subcarrier by motion statistic
        sorted_subcarriers = np.argsort(ms_mean)[::-1]
        smoothed_acf = savgol_filter(self.acf, 5, 3, axis=0)
        

        # get the top 7 subcarriers
        top_subcarriers = sorted_subcarriers[:topK]

        # calculate the combined ACF
        combined_acf = np.mean(smoothed_acf[:, :, top_subcarriers], axis=2)
        # print("combined_acf" , combined_acf.shape)
        # low pass filter   
        # combined_acf = savgol_filter(combined_acf, 5, 3, axis=0)

        # #plot the cominbed ACF
        # plt.figure()
        # plt.plot(self.lag, combined_acf[:, 0])
        # plt.xlabel('Lag (s)')
        # plt.ylabel('Autocorrelation')
        # plt.title('Combined Autocorrelation Function')
        # plt.show()

        # Analyzing peaks for each time slice
        for i in range(len(self.t_s)):
            peaks, _ = find_peaks(combined_acf[:, i], distance=10, prominence=0.05, width=10)
            if peaks.size > 0:
                first_peak = peaks[0]  # Get the first peak
                peak_index.append(first_peak)
                br_t.append(60 / self.lag[first_peak])  # Convert to breathing rate
            else:
                peak_index.append(-1)
                br_t.append(0)

        br_t = np.array(br_t)
        average_br = np.mean(br_t[br_t > 0])  # Calculate average, excluding zeros if any
        
        
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        br_t = np.array(br_t)
        peak_index = np.array(peak_index)
        average_br = np.mean(br_t)
        combined_acf = np.array(combined_acf)
        return combined_acf, peak_index, br_t, average_br
        
    
    def get_presence(self):
        """
        Determines the presence of individuals in an environment based on the autocorrelation function (ACF)
        and average motion statistics (ms_bar).

        This method analyzes ACF and ms_bar data to detect the presence of individuals within different environments.
        The method is designed to work across various datasets without the need for conditional data handling, ensuring
        general applicability. The presence detection algorithm relies on indicators such as motion and breathing rate
        to infer the presence or absence of individuals.

        Returns:
        - bool
            Returns `True` if presence is detected in the environment based on the analysis of the acf and ms_bar data;
            otherwise, `False`.

        """
        presence = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<

        from scipy.signal import peak_prominences
        # Determine presence based on motion statistic and breathing rate
        #Compare the motion statistic againt the peak value as there will be unlikely breathing if the motion level is way larger than the peak value.
        combined_acf, peak_index, br_t, average_br = self.get_br()
        peaks = [combined_acf[peak_index[i], i] for i in range(len(peak_index))]

        # Have high motion stat or have high breathing peaks
        # if (
        #     sum(np.where(self.ms_bar > 0.05, 1, 0)) > len(self.ms_bar) * 0.25
        #     and sum(np.where(peaks / self.ms_bar > 0.15, 1, 0)) > len(self.ms_bar) * 0.25
        # ):
            
        if (
            sum(np.where(self.ms_bar > 0.05, 1, 0)) > len(self.ms_bar) * 0.25 
            and sum(np.where(12 < br_t, 1, 0)) > len(br_t) * 0.25
            and sum(np.where(br_t < 60, 1, 0)) > len(br_t) * 0.25
        ):
            presence = True
        else:
            presence = False

        
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        return presence
    
    
    
    def test_presence(self, data_r="./data/presence", 
                  data_t="train", data_e="env0", 
                  data_p_list=["presence_0.pickle", "presence_1.pickle"],
                  save_p="./results",
                  plot=True):
        figs = []
        for data_p in data_p_list:
            self.data_file_p = osp.join(data_r, data_t, data_e, data_p)
            print(f"Loading data from {self.data_file_p}")
            data_n = data_p.split('.')[0]
            self._load_data(self.data_file_p)
            acf, t_s, lag = self.get_ACF(wl=7, ws=0.2)
            ms, ms_bar = self.get_ms()
            presence = self.get_presence()
            if plot:
                fig_1, ax_1 = self.plot_ms(t_s, ms_bar)
                ax_1.set_title(f"data_n: {data_n} presence: {presence}")
                # Save the figures to one pdf
                figs.append(fig_1)
        if plot:
            os.makedirs(save_p, exist_ok=True)
            fig_p = osp.join(save_p, f'presence_{data_t}_{data_e}_figures.pdf')
            with PdfPages(fig_p) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            for fig in figs:
                plt.close(fig)
        return acf, t_s, lag, ms, ms_bar, presence
    
    def test_br(self, data_r="./data/breath", 
                      data_t = "train", 
                      data_p = "breath_0.pickle", 
                      save_p="./results", 
                      plot=True):
        self.data_file_p = osp.join(data_r, data_t, data_p)
        print(f"Loading data from {self.data_file_p}")
        
        data_n = data_p.split('.')[0]
        self._load_data(self.data_file_p)  # Reload data with the new path
        acf, t_s, lag = self.get_ACF(wl=7, ws=0.2)
        ms, ms_bar = self.get_ms()
        combined_acf, peak_index, br_t, average_br = self.get_br()
        
        if plot:
            fig_1, ax_1 = self.plot_ms(t_s, ms_bar)
            fig_2, ax_2 = self.plot_br(t_s, br_t, average_br)
            fig_3, ax_3 = self.plot_acf_peak_mesh(peak_index, t_s, lag, combined_acf)
            fig_4, ax_4 = self.plot_acf_peak_plot(peak_index, t_s, lag, combined_acf)
            
            # Save the figures to one pdf
            figs = [fig_1, fig_2, fig_3, fig_4]
            os.makedirs(save_p, exist_ok=True)
            fig_p = osp.join(save_p, f'breath_{data_t}_{data_n}_figures.pdf')
            with PdfPages(fig_p) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            
            # close the figures
            plt.close(fig_1)
            plt.close(fig_2)
            plt.close(fig_3)
            plt.close(fig_4)
        return acf, t_s, lag, ms, ms_bar, combined_acf, peak_index, br_t, average_br
    
    @staticmethod
    def plot_ms(t_s, ms_bar):
        fig, ax = plt.subplots(figsize=(10, 5))
        assert len(t_s) == len(ms_bar), 'Time and ms_bar should have the same length'
        ax.plot(t_s, ms_bar, label="MS")
        # set label for ax Motion Statistic
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Motion Statistic')
        return fig, ax
    
    @staticmethod
    def plot_br(t_s, br_t, average_br):
        fig, ax = plt.subplots(figsize=(10, 5),)
        assert len(t_s) == len(br_t), 'Time and br_t should have the same length'
        ax.plot(t_s, br_t, label='Breathing Rate')
        ax.plot(t_s, average_br * np.ones_like(t_s), 'r--', label=f'Average Breathing Rate {average_br:.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Breathing Rate (BPM)')
        ax.legend()
        return fig, ax
    
    @staticmethod
    def plot_acf_peak_mesh(peak_index, t_s, lag, combined_acf):
        fig, ax = plt.subplots(figsize=(10, 5))
        assert len(t_s) == combined_acf.shape[1], 'Time and combined_acf should have the same length'
        assert len(lag) == combined_acf.shape[0], 'Lag and combined_acf should have the same length'
        ax.pcolormesh(t_s, lag[1:], combined_acf[1:, :], cmap='viridis', label='ACF', edgecolors=None)
        ax.plot(t_s, lag[peak_index], 'ro', label='Peak')
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('ACF')
        return fig, ax
    
    @staticmethod
    def plot_acf_peak_plot(peak_index, t_s, lag, combined_acf):
        fig, ax = plt.subplots(figsize=(10, 5))
        assert len(t_s) == combined_acf.shape[1], 'Time and combined_acf should have the same length'
        assert len(lag) == combined_acf.shape[0], 'Lag and combined_acf should have the same length'
        ax.plot(lag, combined_acf)
        ax.plot(lag[peak_index], combined_acf[peak_index], 'ro', label='Peak')
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('ACF')
        return fig, ax


if __name__ == "__main__":
   # You can test your code here
    print(">"*50 + "Testing Breathing" + "<"*50)
    data_r="./data/breath"
    data_t = "train"
    data_p = "breath_2.pickle" 
    save_p = "./results/" # Save to results folder under the current directory (individual_project)
    os.makedirs(save_p, exist_ok=True)
    c = CPD()
    acf, t_s, lag, ms, ms_bar, combined_acf, peak_index, br_t, average_br = c.test_br(data_r=data_r, 
                                                                                      data_t=data_t,
                                                                                      data_p=data_p, 
                                                                                      save_p=save_p,
                                                                                      plot=False)
    
    print(f"ACF: {acf.shape}, t_s: {t_s.shape}, lag: {lag.shape}")
    print(f"MS: {ms.shape}, ms_bar: {ms_bar.shape}")
    print(f"Combined ACF: {combined_acf.shape}, peak_index: {peak_index.shape}, br_t: {br_t.shape}, average_br: {average_br}")
    
    print(">"*50 + "Testing presence detection" + "<"*50)
    data_r = "./data/presence"
    data_t = "train"
    data_e = "env_1"
    data_p_list = ["presence_0.pickle", "presence_1.pickle", "presence_2.pickle"]
    acf, t_s, lag, ms, ms_bar, presence = c.test_presence(data_r=data_r, 
                                                          data_t=data_t, 
                                                          data_e=data_e, 
                                                          data_p_list=data_p_list, 
                                                          save_p=save_p,
                                                          plot=False)
    print(f"ACF: {acf.shape}, t_s: {t_s.shape}, lag: {lag.shape}")
    print(f"MS: {ms.shape}, ms_bar: {ms_bar.shape}")
    print(f"Presence: {presence}")
    
    
    
    
    

    
    
    
    