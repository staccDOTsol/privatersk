from os import listdir
iii = 0
concat_audio = None 
from pydub import AudioSegment

for file in (listdir('./')):
    if '.wav' in file and 'out' in file:
        iii += 1
        try:
            audio = AudioSegment.from_file('out'+str(iii)+'.wav', format="wav")
            if concat_audio is None:
                concat_audio = audio
            else:
                concat_audio += audio
        except:
            print('error')
concat_audio.export("out.wav", format="wav")