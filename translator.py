import os
import sys
import whisper
import numpy as np


from thinkdsp import read_wave
from thinkdsp import Wave
from progress.bar import Bar

from moviepy.editor import VideoFileClip



class Translator:
    def __init__(self, video_path=None):
        '''
        Достает из видео файла аудио и сохраняет его на диск
        достает из файла дорожку, частоту и сохраняет в объект
        video_path - путь до видео файла
        '''
        audio_path = 'audio.wav'
        self.new_sound_signal = np.empty((0, 2), dtype=int)
        self.sound_signal = np.empty((0, 2), dtype=int)
        self.model = whisper.load_model("base")

        # достаем 
        if video_path:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path)
        
        print('Audio reading in progress...')
        wave = read_wave(audio_path)
        os.remove(audio_path)
        print('Audio read successfully.')

        self.framerate = wave.framerate
        self.ys = wave.ys

    def get_sound(self, sound_time=0.1, noise_threshold=0.05):
        '''
        делит дорожку на отрезки длительностью sound_time.
        Полученные отрезки проверяет на порог шума noise_threshold и записывает
        отрезки в self.sound_signal, где был превышен этот порог
        '''
        # длина звука в сэмплах
        len_sig = int(self.framerate * sound_time)

        # делим исходную дорожку на дорожки длиной sound_time
        arr_ys = np.array(np.array_split(self.ys, self.ys.shape[0] // len_sig))

        # в каждом отрезке проверяем, что никакой элемент не превышает по модулю шумовой порог
        # В этом массиве True - отрезок не содержит звука, False - можно считать звуком
        no_signal = np.all((arr_ys < noise_threshold) &
                           (arr_ys > -noise_threshold), axis=1)

        # немного "расширяем" наши дорожки. То есть если в соседний элемент
        # можно считать звуком, то и этот элемент можно считать звуком
        mask = np.logical_or(no_signal[:-2] == False, no_signal[2:] == False)
        no_signal[1:-1][mask] = False

        # найдем где идут переходы со звука на тишину и наоборот
        transition = np.diff(no_signal)
        # получаем массив, где у нас были переходы
        sound_signal = (np.where(transition)[0] + 1) * len_sig
        if not no_signal[0]:
            sound_signal = np.insert(sound_signal, 0, 0)
        if not no_signal[-1]:
            sound_signal = np.append(
                sound_signal, (no_signal.shape[0] - 1) * len_sig)
        # теперь массив состоит из пар вида: номер элемента начала звука, номер конечного элемент звука
        self.sound_signal = np.dstack(
            (sound_signal[::2], sound_signal[1::2] + len_sig))[0]

    def pertion(self):
        '''
        Проходимся по всем сохраненным звуковым отрезкам, полученных в get_sound
        и нарезаем большие куски. Получаем новый массив new_sound_signal
        '''
        for item in self.sound_signal:
            start, stop_tag = item
            self.grinding(self.ys[start:stop_tag], start)

    def grinding(self, audio_ys, start):
        '''
        Нарезает звук на отрезки, пока каждый из них не станет меньшим, чем 2 секунды
        '''
        # получаем громкие звуки
        high_sound = np.concatenate(
            (np.array([0]), np.where(np.abs(audio_ys) > 0.1)[0], audio_ys.shape))

        # делаем массив пар вида: индекс громкого тона, индекс следующего громкого тона,
        sounds = np.dstack([high_sound[:-1], high_sound[1:]])[0]

        # получаем длительность каждого такого отрезка
        durations = sounds[:, 0] - sounds[:, 1]
        # сортируем по длительности (в порядке убывания)
        sorted_durations = np.argsort(durations)
        sounds = sounds[sorted_durations]

        # создаем массив меток, по которым будет происходить нарезка
        tags = np.array([0, audio_ys.shape[0]])
        i = 0
        # пока хотя бы один отрезок из отрезков, разделенных по меткам tags, имеет
        # продолжительность больше, чем 2 сек. будем добавлять метки
        while np.any((tags[1:] - tags[:-1]) > 2 * self.framerate) and i < sounds.shape[0]:
            # добавляем метку, путем деления самого длинного отрезка пополам
            tags = np.append(tags, int(np.mean(sounds[i]) // 1))
            tags = np.sort(tags)
            i += 1

        self.new_sound_signal = np.concatenate(
            (self.new_sound_signal, np.dstack((tags[:-1], tags[1:]))[0] + start))

    def slicing(self):
        '''
        создает нарезку на аудиофайлы и соответствующие им текстовые файлы
        с распознанным текстом
        '''
        # создаем папку
        if not os.path.exists('output'):
            os.makedirs('output')
        # self.file будет отвечать за номер записываемого на диск файла
        self.file = 1
        # метка начала
        start_tag = 0
        # обрабатываемся метка
        dur_tag = 0
        # количество взятых аудио отрезков из self.new_sound_signal
        count = 0
        # индекс обрабатываемого сигнала
        index = 0
        sound_signal = self.new_sound_signal
        self.bar = Bar('Audio file slicing', max=sound_signal.shape[0])
        while dur_tag < self.ys.shape[0] and index < sound_signal.shape[0]:
            if sound_signal[index][0] - start_tag > 5 * self.framerate:
                # случай, когда тихий звук больше 5 секунд
                stop_tag = start_tag + 5 * self.framerate
                self.show(self.ys[start_tag:stop_tag])
                count = 0
                start_tag = stop_tag
                dur_tag = start_tag
                index += 1
                self.bar.next()
            elif sound_signal[index][1] - start_tag <= 5 * self.framerate:
                # случай, когда добавляемый сигнал не превысит длительность 5 секунд
                # записываемого сигнала
                dur_tag = sound_signal[index][1]
                count += 1
                index += 1
                self.bar.next()
            elif sound_signal[index][1] - start_tag > 5 * self.framerate and count > 0:
                # случай, когда следующий добавленный сигнал приведет к тому, что
                # аудио, записываемое на диск будет больше 5 секунд
                stop_tag = dur_tag
                self.show(self.ys[start_tag:stop_tag])
                start_tag = stop_tag
                dur_tag = start_tag
                count = 0
                index += 1
                self.bar.next()
            else:
                # случай, когда звук из self.new_sound_signal все равно остался
                # длинным (исключает зацикливание на некоторых сложных входных данных)
                stop_tag = start_tag + 5 * self.framerate
                self.show(self.ys[start_tag:stop_tag])
                count = 0
                start_tag = stop_tag
                dur_tag = start_tag
        self.show(self.ys[start_tag:self.ys.shape[0]])
        self.bar.finish()

        # Проходимся по всем аудио файлам и создаем текстовые файлы с распознанным текстом
        self.bar = Bar('Extracting text from audio', max=self.file)
        for i in range(1, self.file):
            result = self.model.transcribe(f'output/audio_{i}.wav')
            with open(f'output/text_{i}.txt', 'w') as file:
                file.write(result["text"])
            self.bar.next()
        self.bar.next()
        self.bar.finish()

    def show(self, ys):
        '''
        записывает на диск аудио файл с волной ys
        '''
        wave = Wave(ys, framerate=self.framerate)
        wave.write(f'output/audio_{self.file}.wav')
        self.file += 1
        
if len(sys.argv) < 2:
    print("Usage: python translator.py input_video_path")
else:
    # создаем объект
    translator = Translator(sys.argv[1])

    # делаем "грубую" нарезку
    translator.get_sound()

    # делаем более точную нарезку
    translator.pertion()

    # создаем нарезанные аудио файлы и соответствующие им текстовые файлы
    # с распознанным текстом
    translator.slicing()
