import sounddevice as sd

def play_signal(signal, fs):
    sd.play(signal, fs)
    sd.wait()