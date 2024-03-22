import pypinyin
from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin.contrib.tone_convert as tone_convert
from pypinyin.contrib.tone_convert import to_normal, to_tone, to_initials, to_finals
from typing import List, Dict

def note_num_to_note_name(note_num):
    if note_num == 'rest':
        return note_num
    octave = int(note_num) // 12 - 1
    note = int(note_num) % 12
    note_string = [
        f'C{octave}',
        f'C#{octave}/Db{octave}',
        f'D{octave}',
        f'D#{octave}/Eb{octave}',
        f'E{octave}',
        f'F{octave}',
        f'F#{octave}/Gb{octave}',
        f'G{octave}',
        f'G#{octave}/Ab{octave}',
        f'A{octave}',
        f'A#{octave}/Bb{octave}',
        f'B{octave}'
    ]
    return note_string[note]


def split_word_to_phoneme(data):

    def split_item(item):
        if item['lyric'] in ['AP', 'SP']:
            return [{
                'dur': item['duration'],
                'lyric': item['lyric'],
                'noteNumber': note_num_to_note_name(item['noteNumber']),
                'phone': item['lyric'],
                'gtdur': item['duration'],
            }]
        pinyin = pypinyin.lazy_pinyin(item['lyric'])[0]
        ret = []
        initial = tone_convert.to_initials(pinyin)
        final = tone_convert.to_finals(pinyin)
        if (initial != ''):
            ret.append({
                'dur': item['duration'],
                'lyric': item['lyric'],
                'noteNumber': note_num_to_note_name(item['noteNumber']),
                'phone': initial,
                'gtdur': item['duration'],
            })
        if (final != ''):
            ret.append({
                'dur': item['duration'],
                'lyric': item['lyric'],
                'noteNumber': note_num_to_note_name(item['noteNumber']),
                'phone': final,
                'gtdur': item['duration'],
            })
        return ret

    ret = [splitted_item for item in data for splitted_item in split_item(item)]
    return ret


def phoneme_items_to_string(data):
    fileid = 2023102320
    lyric = ''.join([item['lyric'] for item in data if item['lyric'] not in ['AP', 'SP']])
    phone = ' '.join([item['phone'] for item in data])
    noteNumber = ' '.join([str(item['noteNumber']) for item in data])
    dur = ' '.join([str(item['dur']) for item in data])
    gtdur = ' '.join([str(item['gtdur']) for item in data])
    slur = ' '.join(['0' for item in data])
    return f"{fileid}|{lyric}|{phone}|{noteNumber}|{dur}|{gtdur}|{slur}"

def tick_to_second(tick, tempo, resolution):
    return tick * 60 / (tempo * resolution)



def convert_data_tick_to_second(data: List[dict], tempo, resolution):
    return [
        {
            **item,
            'time': tick_to_second(item['time'], tempo, resolution),
            'duration': tick_to_second(item['duration'], tempo, resolution),
        }
        for item in data
    ]


def fill_gaps(events, starting_point=0):
    if len(events) == 0:
        return []

    ret = []
    if events[0]['time'] > starting_point:
        ret.append({
            'time': starting_point,
            'duration': events[0]['time'] - starting_point,
            'lyric': 'AP',
            'noteNumber': 'rest',
        })
        starting_point = events[0]['time']
    index = 0
    while index < len(events):
        if events[index]['time'] > starting_point:
            ret.append({
                'time': starting_point,
                'duration': events[index]['time'] - starting_point,
                'lyric': 'AP',
                'noteNumber': 'rest',
            })
            starting_point = events[index]['time']
        else:
            ret.append({
                'time': events[index]['time'],
                'duration': events[index]['duration'],
                'lyric': events[index]['lyric'],
                'noteNumber': events[index]['noteNumber'],
            })
            starting_point = events[index]['time'] + events[index]['duration']
            index += 1
    return ret

