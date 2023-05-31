import os
import muspy

def merge_tracks(
	input_path: str,
	song_name: str,
	output_path: str
):
	files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.split(".")[-1] =="mid"]
	if len(files) < 2:
		return

	first_instrument = muspy.read_midi(files[0])
	song = muspy.Music(
		metadata=muspy.Metadata(title=song_name), 
		resolution=first_instrument.resolution, 
		tempos=first_instrument.tempos, 
		key_signatures=first_instrument.key_signatures, 
		time_signatures=first_instrument.time_signatures, 
		beats=first_instrument.beats, 
		lyrics=None, 
		annotations=None, 
		tracks=[]
	)

	for f in files:
		track = muspy.read_midi(f).tracks[0]
		track.name = f.split(".")[0]
		if "drum" in f.lower():
			track.is_drum = True
		song.tracks.append(track)

	song.write(os.path.join(output_path, song_name+".mid"))

