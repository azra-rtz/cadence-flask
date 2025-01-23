from flask import Flask, request, jsonify
from mido import MidiFile, MidiTrack, Message, bpm2tempo, MetaMessage
import os
import json
import logging
import traceback
import numpy as np
import soundfile as sf
import aubio
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from pydub import AudioSegment
import tempfile
from client import supabase
from datetime import datetime

app = Flask(__name__)

# Configuration
PITCH_DIR = Path('./pitch')
DATA_JSON_PATH = PITCH_DIR / 'data.json'
SAMPLE_RATE = 44100
WINDOW_SIZE = 4096
HOP_SIZE = WINDOW_SIZE // 2

# Ensure the pitch directory exists
PITCH_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_midi(data: Dict[str, Any], ticks_per_beat: int = 480, tempo_bpm: int = 120) -> str:
    """
    Generate a MIDI file from the provided note data.
    
    Args:
        data: Dictionary containing notes and title information
        ticks_per_beat: MIDI ticks per beat
        tempo_bpm: Tempo in beats per minute
    
    Returns:
        str: Generated MIDI filename
    
    Raises:
        Exception: If MIDI generation fails
    """
    try:
        logger.debug(f"Starting MIDI generation with data: {json.dumps(data, indent=2)}")
        
        tempo = bpm2tempo(tempo_bpm)
        midi = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        midi.tracks.append(track)
        
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        
        seconds_per_beat = tempo / 1_000_000
        seconds_per_tick = seconds_per_beat / ticks_per_beat
        prev_time = 0
        
        for note in data["notes"]:
            logger.debug(f"Processing note: {note}")
            midi_note = note["midi"]
            start_time = float(note["time"])
            duration = float(note["duration"])
            
            start_ticks = int(start_time / seconds_per_tick)
            duration_ticks = int(duration / seconds_per_tick)
            delta_time = start_ticks - prev_time
            
            if midi_note >= 0 and duration_ticks > 0:
                track.append(Message('note_on', channel=0, note=int(midi_note), velocity=64, time=delta_time))
                track.append(Message('note_off', channel=0, note=int(midi_note), velocity=64, time=duration_ticks))
            
            prev_time = start_ticks
        
        output_filename = f"{data['title']}.mid"
        output_path = PITCH_DIR / output_filename
        logger.debug(f"Saving MIDI file to: {output_path}")
        
        midi.save(str(output_path))
        logger.debug("MIDI file saved successfully")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"Error in generate_midi: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def detect_pitch(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Detect pitch in an audio file using aubio.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        List of dictionaries containing pitch data with time and duration
    
    Raises:
        ValueError: If sample rate doesn't match expected rate
    """
    pitch_o = aubio.pitch("yin", WINDOW_SIZE, HOP_SIZE, SAMPLE_RATE)
    pitch_o.set_unit("midi")

    audio_data, sr = sf.read(str(file_path))

    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate of audio file ({sr}) does not match the expected rate ({SAMPLE_RATE})")

    pitch_data = []
    last_pitch_val: Optional[float] = None
    note_start_time: Optional[float] = None
    note_end_time: Optional[float] = None

    for i in range(0, len(audio_data), HOP_SIZE):
        samples = audio_data[i:i + HOP_SIZE]
        if len(samples) < HOP_SIZE:
            continue

        if samples.ndim > 1:
            samples = samples[:, 0]
        samples = samples.astype(np.float32)

        pitch_val = pitch_o(samples)[0]
        timestamp = i / SAMPLE_RATE

        if pitch_val > 0:
            if pitch_val != last_pitch_val:
                if last_pitch_val is not None and note_start_time is not None:
                    end_time = note_end_time if note_end_time is not None else timestamp
                    pitch_data.append({
                        "midi": int(last_pitch_val),
                        "time": note_start_time,
                        "duration": end_time - note_start_time
                    })
                note_start_time = timestamp
            last_pitch_val = pitch_val
            note_end_time = None
        elif last_pitch_val is not None and note_start_time is not None:
            note_end_time = timestamp
            pitch_data.append({
                "midi": int(last_pitch_val),
                "time": note_start_time,
                "duration": note_end_time - note_start_time
            })
            last_pitch_val = None
            note_start_time = None

    if last_pitch_val is not None and note_start_time is not None:
        note_end_time = note_end_time or timestamp
        pitch_data.append({
            "midi": int(last_pitch_val),
            "time": note_start_time,
            "duration": note_end_time - note_start_time
        })

    return combine_pitch_data(pitch_data)

def combine_pitch_data(pitch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine consecutive identical notes and round their timings.
    
    Args:
        pitch_data: List of pitch data dictionaries
    
    Returns:
        List of combined pitch data dictionaries
    """
    combined_pitch_data = []
    previous_note = None

    for note in pitch_data:
        if previous_note and previous_note["midi"] == note["midi"]:
            previous_note["duration"] += note["duration"]
        else:
            if previous_note:
                combined_pitch_data.append(previous_note)
            previous_note = note

    if previous_note:
        combined_pitch_data.append(previous_note)

    for note in combined_pitch_data:
        note["time"] = round(note["time"], 1)
        note["duration"] = round(note["duration"], 1)

    return combined_pitch_data

@app.route('/generate-midi', methods=['POST'])
def create_midi():
    """Handle MIDI generation endpoint."""
    try:
        logger.debug("Received /generate-midi request")
        data = request.json
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data or 'title' not in data or 'notes' not in data:
            return jsonify({
                "message": "Missing required data",
                "received": data
            }), 400
        
        try:
            filename = generate_midi(data)
            file_path = PITCH_DIR / filename
            
            if not file_path.exists():
                return jsonify({"message": "Generated file not found"}), 500
            
            with open(file_path, 'rb') as file:
                midi_data = file.read()
            
            import base64
            midi_base64 = base64.b64encode(midi_data).decode('utf-8')
            
            os.remove(file_path)

            return jsonify({
                "success": True,
                "filename": filename,
                "data": midi_base64
            })
            
        except Exception as e:
            logger.error(f"Error in MIDI generation: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "message": "Error generating MIDI file",
                "error": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error in request handling: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Error processing request",
            "error": str(e)
        }), 500

def is_wav_file(file_path: Path) -> bool:
    """
    Check if the file is already a WAV file with correct parameters.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        bool: True if file is WAV with correct parameters, False otherwise
    """
    try:
        if file_path.suffix.lower() != '.wav':
            return False
            
        # Check WAV file parameters using soundfile
        info = sf.info(str(file_path))
        
        # Check if the WAV file meets our requirements
        return (
            info.samplerate == SAMPLE_RATE and
            info.channels == 1  # Mono audio
        )
    except Exception as e:
        logger.error(f"Error checking WAV file: {str(e)}")
        return False

def convert_to_wav(input_file: Path) -> Path:
    """
    Convert any audio file to WAV format if needed.
    
    Args:
        input_file: Path to the input audio file
        
    Returns:
        Path to the WAV file (either converted or original)
        
    Raises:
        ValueError: If audio conversion fails
    """
    try:
        # Check if input file exists
        if not input_file.exists():
            raise ValueError(f"Input file does not exist: {input_file}")
            
        # If file is already a compatible WAV file, return it directly
        if is_wav_file(input_file):
            logger.info("File is already a compatible WAV file, using as-is")
            return input_file
            
        logger.info(f"Converting {input_file.suffix} file to WAV format")
        
        # Create a temporary file for the WAV output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = Path(temp_wav.name)
        
        # Load the audio file using pydub
        try:
            audio = AudioSegment.from_file(str(input_file))
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")
        
        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        # Set sample rate if needed
        if audio.frame_rate != SAMPLE_RATE:
            audio = audio.set_frame_rate(SAMPLE_RATE)
        
        # Export as WAV
        try:
            audio.export(
                str(temp_wav_path),
                format='wav',
                parameters=[
                    '-ar', str(SAMPLE_RATE),
                    '-ac', '1'  # Ensure mono
                ]
            )
        except Exception as e:
            # Clean up temp file if export fails
            if temp_wav_path.exists():
                temp_wav_path.unlink()
            raise ValueError(f"Failed to export WAV file: {str(e)}")
            
        logger.info(f"Successfully converted to WAV: {temp_wav_path}")
        return temp_wav_path
        
    except Exception as e:
        logger.error(f"Error in convert_to_wav: {str(e)}")
        raise

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    """Handle audio upload and pitch detection endpoint."""
    try:
        logger.info("=== Starting audio upload processing ===")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Form data keys: {list(request.form.keys())}")
        logger.info(f"Files keys: {list(request.files.keys())}")

        # Validate request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({"message": "No audio file provided"}), 400

        username = request.form.get('username')
        title = request.form.get('title')
        audio_file = request.files['audio']

        if not all([username, title, audio_file]):
            logger.error(f"Missing required fields: username={username}, title={title}, audio_file={bool(audio_file)}")
            return jsonify({"message": "Missing required fields"}), 400

        if not audio_file.filename:
            logger.error("Audio file has no filename")
            return jsonify({"message": "Invalid audio file"}), 400

        # Original filename
        filename = Path(audio_file.filename).name

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Append the timestamp to the filename to make it unique
        secure_filename = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
        print("DWAIHJDWIOADJWAIODJWAIDJAWDOAWJOIADJWAODJAWIODJAW", secure_filename)
        original_file_path = PITCH_DIR / secure_filename
                
        # Ensure directory exists
        PITCH_DIR.mkdir(exist_ok=True)
        
        try:
            # Save uploaded file
            audio_file.save(str(original_file_path))
            
            with open(str(original_file_path), 'rb') as file:
                # Upload to Supabase storage
                response = supabase.storage.from_("audios").upload(
                    path=secure_filename,  # Use the secure filename as the path
                    file=file,
                    file_options={
                        "cache-control": "3600",
                        "upsert": False  # Use boolean instead of string
                    }
                )
            
            # # Delete the local file after upload
            # os.remove(str(original_file_path))
            # Get the list of files (optional)
            file_list = supabase.storage.from_("audios").list()
            
            # Get the public URL of the uploaded file
            file_url = supabase.storage.from_("audios").get_public_url(secure_filename)
            
            print("Upload Response:", response)
            print("File List:", file_list)
            print("File URL:", file_url)
            logger.info(f"File saved successfully: {original_file_path}")
            
            if not original_file_path.exists():
                raise ValueError("File was not saved successfully")

            # Convert to WAV if needed
            wav_file_path = convert_to_wav(original_file_path)
            logger.info(f"WAV conversion complete: {wav_file_path}")

            # Detect pitch
            logger.info("Starting pitch detection...")
            detected_pitches = detect_pitch(wav_file_path)
            logger.info(f"Detected {len(detected_pitches)} pitches")

            new_transcription = {
                "title": title,
                "audio_title": secure_filename,
                "notes": detected_pitches
            }

                # Insert transcript and get response
            response = supabase.table('transcriptions').insert({"username": username, "data": new_transcription}).execute()
            response = supabase.table('transcriptions').select().execute()
            print("Data: ", response.data)

            logger.info("Transcription saved successfully")
            
            return jsonify({
                "message": "Audio uploaded and processed successfully!",
                "transcription": new_transcription
            })

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            logger.error(traceback.format_exc())
            # Cleanup any partial files
            if original_file_path.exists():
                original_file_path.unlink()
            if 'wav_file_path' in locals() and wav_file_path != original_file_path and wav_file_path.exists():
                wav_file_path.unlink()
            raise

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Error processing upload",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/transcriptions', methods=['GET'])
def get_transcriptions():
    """Get all transcriptions endpoint."""
    try:
        response = supabase.table('transcriptions').select("*").execute()
        
        if not response.data:
            return jsonify([]), 200
        
        # Transform data into the desired format
        transformed_data = {}
        for item in response.data:
            username = item["username"]
            data = item["data"]
            if username not in transformed_data:
                transformed_data[username] = []  # Initialize with an empty list
            transformed_data[username].append(data)
        
        return jsonify(transformed_data), 200
    except Exception as e:
        return jsonify({"message": f"Error fetching transcriptions: {str(e)}"}), 5004

@app.route('/transcriptions/<username>/<audio_title>', methods=['DELETE'])
def delete_transcription(username: str, audio_title: str):
    """Delete a specific transcription endpoint."""
    try:
        response = supabase.table('transcriptions').select().execute()
        # Transform data into the desired format
        transformed_data = {}
        for item in response.data:
            username = item["username"]
            data = item["data"]
            if username not in transformed_data:
                transformed_data[username] = []  # Initialize with an empty list
            transformed_data[username].append({"id": item["id"], "data": data})

        # print(transformed_data)
        for detail in transformed_data[username]:
            # print(detail['data']['audio_title'], audio_title)
            if detail['data']['audio_title'] == audio_title:
                response = supabase.table('transcriptions').delete().eq("id", detail['id']).execute()
                return jsonify({"message": "Transcription and associated audio file deleted successfully"}), 200
        return jsonify({"message": f"Error deleting transcription"}), 500
    except Exception as e:
        return jsonify({"message": f"Error deleting transcription: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()