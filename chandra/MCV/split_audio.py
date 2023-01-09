import argparse
import collections
import contextlib
import os
import sys
import wave
from pathlib import Path

import webrtcvad


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    # num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    # ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    # triggered = False

    # voiced_frames = []

    total_frames = len(frames)
    # print("WHATEVER")

    lst = []
    index = 0
    while index < total_frames:
        frame = frames[index]
        is_voiced = vad.is_speech(frame.bytes, sample_rate)
        if is_voiced:
            lst.append(frame)
        else:
            if lst:
                yield b"".join([f.bytes for f in lst])
                lst = []
        index += 1
    if lst:
        yield b"".join([f.bytes for f in lst])
        lst = []

    # frames_per_audio = 20
    # start = 2
    # while start < total_frames:
    #     lst = frames[start - 2 : start + frames_per_audio + 2]
    #     status = [vad.is_speech(frame.bytes, sample_rate) for frame in lst]
    #     num_voiced = len([1 for _ in status if _])
    #     num_unvoiced = len([1 for _ in status if not _])
    #     if num_unvoiced < 0.2 * len(status):
    #         start += frames_per_audio
    #         yield b"".join([f.bytes for f in lst])
    #     else:
    #         print(status)
    #         start += 2

    # for frame_num, frame in enumerate(frames):
    #     print(int(vad.is_speech(frame.bytes, sample_rate)), end="")

    # for frame_num, frame in enumerate(frames):
    #     is_speech = vad.is_speech(frame.bytes, sample_rate)

    #     sys.stdout.write("1" if is_speech else "0")
    #     if not triggered:
    #         ring_buffer.append((frame, is_speech))
    #         num_voiced = len([f for f, speech in ring_buffer if speech])
    #         # If we're NOTTRIGGERED and more than 90% of the frames in
    #         # the ring buffer are voiced frames, then enter the
    #         # TRIGGERED state.
    #         if num_voiced > 0.9 * ring_buffer.maxlen:
    #             triggered = True
    #             sys.stdout.write("+(%s)" % (ring_buffer[0][0].timestamp,))
    #             # We want to yield all the audio we see from now until
    #             # we are NOTTRIGGERED, but we have to start with the
    #             # audio that's already in the ring buffer.
    #             for f, s in ring_buffer:
    #                 voiced_frames.append(f)
    #             ring_buffer.clear()
    #     else:
    #         # We're in the TRIGGERED state, so collect the audio data
    #         # and add it to the ring buffer.
    #         voiced_frames.append(frame)
    #         ring_buffer.append((frame, is_speech))
    #         num_unvoiced = len([f for f, speech in ring_buffer if not speech])
    #         # If more than 90% of the frames in the ring buffer are
    #         # unvoiced, then enter NOTTRIGGERED and yield whatever
    #         # audio we've collected.
    #         if (
    #             num_unvoiced
    #             > 3
    #             # and len(voiced_frames) > 120
    #             # and total_frames - frame_num > 70
    #         ):
    #             sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
    #             triggered = False
    #             yield b"".join([f.bytes for f in voiced_frames])
    #             ring_buffer.clear()
    #             voiced_frames = []
    # if triggered:
    #     sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
    # sys.stdout.write("\n")
    # # If we have any leftover voiced audio when we run out of input,
    # # yield it.
    # if voiced_frames:
    #     yield b"".join([f.bytes for f in voiced_frames])


def main(args):
    print(args)
    #     parser = argparse.ArgumentParser(help="Python file for Voice Activity Detection based splitting")
    #     parser.add_argument("--aggressiveness", type=int, required=True, choices=[0,1,2,3])
    #     parser.add_argument("--inp_wav_path", type=str, required=True)
    #     parser.add_argument("--outp_wav_path", type=str, required=True)
    #     args = vars(parser.parse_args())
    agg = args["aggressiveness"]
    inp_wav_path = args["inp_wav_path"]
    outp_wav_path = args["outp_wav_path"]
    frame_duration = args["frame_duration"]
    padding_duration = args["padding_duration"]
    audio, sample_rate = read_wave(inp_wav_path)
    vad = webrtcvad.Vad(agg)
    print("WHO IS THE PERSON")
    frames = frame_generator(frame_duration, audio, sample_rate)
    print("WHO IS THE FRAME")
    frames = list(frames)
    segments = vad_collector(sample_rate, frame_duration, padding_duration, vad, frames)
    print("WHO IS THE SEGMENTS")
    for i, segment in enumerate(segments):
        path = f"{outp_wav_path}" + "-chunk-%002d.wav" % (i,)
        print(" Writing %s" % (path,))
        write_wave(path, segment, sample_rate)


if __name__ == "__main__":

    data_dir = os.path.join(
        "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/chandra/MCV/data/audio_files"
    )
    all_paths = list([p for p in Path(data_dir).glob("*.wav")])
    output_dir = os.path.join(
        "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/chandra/MCV/data/audio_files"
    )
    os.makedirs(output_dir, exist_ok=True)
    # print("data_dir: ", data_dir)
    # print("output_dir: ", output_dir)
    # print(all_paths)

    for aggression in range(4):
        for path in all_paths:
            par_dir_name, file_name = os.path.split(path)
            remaining_dir = par_dir_name.replace(data_dir, "")
            remaining_dir = remaining_dir.strip("/")

            output_file_dir = os.path.join(
                output_dir, f"mod-agg-{aggression}", remaining_dir
            )
            os.makedirs(output_file_dir, exist_ok=True)

            main(
                {
                    "aggressiveness": aggression,
                    "inp_wav_path": str(path),
                    "outp_wav_path": str(
                        os.path.join(output_file_dir, file_name.replace(".wav", ""))
                    ),
                    "frame_duration": 10,
                    "padding_duration": 100,
                }
            )
