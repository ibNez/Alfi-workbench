import pyttsx3
print("hello")
engine = pyttsx3.init()

voices = engine.getProperty("voices")
for single_voice in voices:
    print("recoded: " + single_voice.name)
    if 'english-us' in single_voice.name:
        single_voice_name = single_voice.name
        print(single_voice.id)
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate+10)
        engine.setProperty("voice",single_voice.id)
        engine.save_to_file("This is a test of voice "+single_voice_name, 'voiceid'+single_voice.id+'.wav')
        engine.runAndWait()