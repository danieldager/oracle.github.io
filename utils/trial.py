import os
import keyboard
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time, sleep
from datetime import datetime

from pyplr.pupil import PupilCore
from pyplr.utils import unpack_data_pandas


""" notes """


"""                code to open Pupil Capture
sudo /Applications/Pupil\ Capture.app/Contents/MacOS/pupil_capture
open /usr/local/opt/labrecorder/LabRecorder/LabRecorder.app
open /opt/homebrew/Cellar/labrecorder/1.16.2_9/LabRecorder/LabRecorder.app
"""


""" setup """

p = PupilCore()

# these are the directories where the data will be saved
trials_dir = "trials"
pupil_data_dir = "pupil_data"


def run_trials(n_trials, min_time, max_time):
            
    for i in range(n_trials):
        print(f'\nTrial {i + 1}')

        # select a random trial duration between 60 and 180 seconds
        trial_duration = np.random.randint(min_time, max_time)

        # create a unique trial name, to be used as filename for data
        trial_name = datetime.now().strftime('%m%d-%H%M')

        # begin recording pupil diameter data
        pgr_future = p.pupil_grabber(topic='pupil.1.3d', seconds=trial_duration + 1)


        """ grey screen """
        # https://www.fullscreen.one/
        os.system('open grey.png')


        """ track keypresses """

        sequence = []
        k_times = []
        start = time()

        def on_key_event(event):
            if event.event_type == "down":
                if event.name == "left":
                    # print(0)
                    sequence.append(0)
                    k_times.append(time() - start)
                elif event.name == "right":
                    # print(1)
                    sequence.append(1)
                    k_times.append(time() - start)

        def track_keypresses():
            keyboard.hook(on_key_event)
            sleep(trial_duration)
            keyboard.unhook_all()
            # os.system('osascript -e \'quit app "Preview"\'')
            # os.remove(grey_image_path)

        keypress_thread = threading.Thread(target=track_keypresses)
        keypress_thread.start()
        keypress_thread.join()


        """ post-processing """

        # convert python lists into numpy arrays
        sequence = np.array(sequence)

        # round to 2 decimal places to sync with pupil data
        k_times = np.array(k_times).round(2)

        # delays between keypresses, prepend 0 for first keypress
        delays = np.hstack((0, np.diff(k_times)))

        # extract pupil data
        pupil_data = pgr_future.result()
        pupil_data = unpack_data_pandas(pupil_data, cols=['diameter_3d'])

        # save pupil data to csv
        pupil_data.to_csv(os.path.join(pupil_data_dir, trial_name + ".csv"), index=False)

        # extract pupil diameter at keypress times
        start = pupil_data.index[0]
        p_times = (pupil_data.index - start).round(2)

        # grab only the index of the pupil diameter closest to the keypress time
        indices = []
        for k_time in k_times:
            index = np.argmin(np.abs(p_times - k_time))
            indices.append(index)

        diameters = [pupil_data['diameter_3d'].iloc[i] for i in indices]

        # create array to be exported to RNN/LSTM model
        features = np.vstack((sequence, delays, diameters))
        np.save(os.path.join(trials_dir, trial_name), features)


if __name__ == "__main__":
    run_trials(10, 30, 80)
    print('\nAll trials complete!')