from dataset.signal import generate_signal, SAMPLE_RATE
from util.audio import play_signal

def testC34():
    signal = generate_signal(1.50, ['C3', 'C4'])
    play_signal(signal, SAMPLE_RATE)


if __name__ == "__main__":
    testC34()