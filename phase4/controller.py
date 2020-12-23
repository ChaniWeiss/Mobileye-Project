import pickle
from phase4.data_holder import DataHolder
from phase4.tflman import TflMan
from tensorflow.python.keras.models import load_model


class Controller:
    @staticmethod
    def load_pkl_data(pkl_path, frames):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        focal = data['flx']
        pp = data['principle_point']
        id = int(frames[0][-23:-17])
        EM = [data['egomotion_' + str(id + i) + '-' + str(id + 1 + i)] for i in range(len(frames) - 1)]
        return focal, pp, EM

    @staticmethod
    def load_pls_data(pls_path):
        with open(pls_path, "r") as file:
            data = file.readlines()
        pkl_file = data[0]
        frames = data[1:]
        return frames, pkl_file

    def run(self, pls_path):
        loaded_model = load_model('C:\mobileye\\model.h5')
        frames, pkl_path = Controller.load_pls_data(pls_path)
        focal, pp, EMs = Controller.load_pkl_data(pkl_path[:-1], frames)
        dh = DataHolder(pp, focal)
        tfl_manager = TflMan()

        for i, frame in enumerate(frames[:-1]):
            dh.EM = EMs[i]
            yield tfl_manager.run_on_frame(frame[:-1], dh, loaded_model)
