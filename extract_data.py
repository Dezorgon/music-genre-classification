import librosa
import json
import os


def preprocess_data(source_data_path, mfcc=False, melspec=False, sr=22050, duration=30,
                    num_spec_coefficient=13, n_fft=2048, hop_length=512, num_segments=10) -> dict:
    if sum([mfcc, melspec]) != 1:
        raise ValueError()

    data = {
        "data": [],
        "labels": [],
    }

    samples_per_segment = int((sr * duration) / num_segments)

    for i, (directory, _, files) in enumerate(os.walk(source_data_path)):
        if directory is not source_data_path:
            for f in files:
                try:
                    file_path = os.path.join(directory, f)
                    signal, sample_rate = librosa.load(file_path, sr=sr)
                except Exception as e:
                    print(e)
                else:
                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        if len(signal) < finish:
                            continue

                        spectrogram = []

                        if mfcc:
                            spectrogram = librosa.feature.mfcc(signal[start:finish], sample_rate,
                                                               n_mfcc=num_spec_coefficient,
                                                               n_fft=n_fft, hop_length=hop_length)
                        if melspec:
                            mfcc = librosa.feature.melspectrogram(signal[start:finish], sample_rate,
                                                                  n_mels=num_spec_coefficient,
                                                                  n_fft=n_fft, hop_length=hop_length)
                            mfcc = librosa.power_to_db(mfcc ** 2)

                        spectrogram = spectrogram.T

                        data["data"].append(spectrogram.tolist())
                        data["labels"].append(i - 1)

    return data


if __name__ == "__main__":
    data = preprocess_data(r"D:\Downloads\Data\genres_original", mfcc=True)

    with open("data.json", "w") as fp:
        json.dump(data, fp, indent=4)

