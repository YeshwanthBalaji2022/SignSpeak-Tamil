from flask import Flask, render_template, Response, jsonify
from gtts import gTTS
import warnings
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import io
import scipy.io.wavfile as wav
import numpy as np
from IPython.display import Audio
import pygame
import threading  # Import threading for lock

app = Flask(__name__)
final_word=""
word_lock = threading.Lock()

DEFAULT_SAMPLING_RATE = 16000

def text_to_speech(text, lang='ta'):
    if not text.strip():
        return

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        byte_io = io.BytesIO()
        tts.write_to_fp(byte_io)
        byte_io.seek(0)

        pygame.mixer.init(frequency=DEFAULT_SAMPLING_RATE)
        pygame.mixer.music.load(byte_io, 'mp3')
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(1)
    except Exception as e:
        print(f"An error occurred: {e}")

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

def util(hand_predictions):
    true_dict = {
        'அ': 'a', 'ஆ': 'ā', 'இ': 'i', 'ஈ': 'ī', 'உ': 'u', 'ஊ': 'ū', 'எ': 'e', 'ஏ': 'ē', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'ō', 'ஔ': 'au', 'ஃ': 'ak',
        'க்': 'k', 'ங்': 'ṅ', 'ச்': 'c', 'ஞ்': 'ñ', 'ட்': 'ṭ', 'ண்': 'ṇ', 'த்': 't', 'ந்': 'n', 'ப்': 'p', 'ம்': 'm', 'ய்': 'y', 'ர்': 'r', 'ல்': 'l',
        'வ்': 'v', 'ழ்': 'lzh', 'ள்': 'll', 'ற்': 'ṟ', 'ன்': 'ṉ', 'க': 'ka', 'கா': 'kā', 'கி': 'ki', 'கீ': 'kī', 'கு': 'ku', 'கூ': 'kū', 'கெ': 'ke',
        'கே': 'kē', 'கை': 'kai', 'கொ': 'ko', 'கோ': 'kō', 'கௌ': 'kau', 'ங': 'nga', 'ஙா': 'ngā', 'ஙி': 'ngi', 'ஙீ': 'ngī', 'ஙு': 'ngu', 'ஙூ': 'ngū',
        'ஙெ': 'nge', 'ஙே': 'ngē', 'ஙை': 'ngai', 'ஙொ': 'ngo', 'ஙோ': 'ngō', 'ஙௌ': 'ngau', 'ச': 'sa', 'சா': 'sā', 'சி': 'si', 'சீ': 'sī', 'சு': 'su',
        'சூ': 'sū', 'செ': 'se', 'சே': 'sē', 'சை': 'sai', 'சொ': 'so', 'சோ': 'sō', 'சௌ': 'sau', 'ஞ': 'ña', 'ஞா': 'ñā', 'ஞி': 'ñi', 'ஞீ': 'ñī',
        'ஞு': 'ñu', 'ஞூ': 'ñū', 'ஞெ': 'ñe', 'ஞே': 'ñē', 'ஞை': 'ñai', 'ஞொ': 'ño', 'ஞோ': 'ñō', 'ஞௌ': 'ñau', 'ட': 'ṭa', 'டா': 'ṭā', 'டி': 'ṭi',
        'டீ': 'ṭī', 'டு': 'ṭu', 'டூ': 'ṭū', 'டெ': 'ṭe', 'டே': 'ṭē', 'டை': 'ṭai', 'டொ': 'ṭo', 'டோ': 'ṭō', 'டௌ': 'ṭau', 'ண': 'ṇa', 'ணா': 'ṇā',
        'ணி': 'ṇi', 'ணீ': 'ṇī', 'ணு': 'ṇu', 'ணூ': 'ṇū', 'ணெ': 'ṇe', 'ணே': 'ṇē', 'ணை': 'ṇai', 'ணொ': 'ṇo', 'ணோ': 'ṇō', 'ணௌ': 'ṇau', 'த': 'ta',
        'தா': 'tā', 'தி': 'ti', 'தீ': 'tī', 'து': 'tu', 'தூ': 'tū', 'தெ': 'te', 'தே': 'tē', 'தை': 'tai', 'தொ': 'to', 'தோ': 'tō', 'தௌ': 'tau',
        'ந': 'na', 'நா': 'nā', 'நி': 'ni', 'நீ': 'nī', 'நு': 'nu', 'நூ': 'nū', 'நெ': 'ne', 'நே': 'nē', 'நை': 'nai', 'நொ': 'no', 'நோ': 'nō',
        'நௌ': 'nau', 'ப': 'pa', 'பா': 'pā', 'பி': 'pi', 'பீ': 'pī', 'பு': 'pu', 'பூ': 'pū', 'பெ': 'pe', 'பே': 'pē', 'பை': 'pai', 'பொ': 'po',
        'போ': 'pō', 'பௌ': 'pau', 'ம': 'ma', 'மா': 'mā', 'மி': 'mi', 'மீ': 'mī', 'மு': 'mu', 'மூ': 'mū', 'மெ': 'me', 'மே': 'mē', 'மை': 'mai',
        'மொ': 'mo', 'மோ': 'mō', 'மௌ': 'mau', 'ய': 'ya', 'யா': 'yā', 'யி': 'yi', 'யீ': 'yī', 'யு': 'yu', 'யூ': 'yū', 'யெ': 'ye', 'யே': 'yē',
        'யை': 'yai', 'யொ': 'yo', 'யோ': 'yō', 'யௌ': 'yau', 'ர': 'ra', 'ரா': 'rā', 'ரி': 'ri', 'ரீ': 'rī', 'ரு': 'ru', 'ரூ': 'rū', 'ரெ': 're',
        'ரே': 'rē', 'ரை': 'rai', 'ரொ': 'ro', 'ரோ': 'rō', 'ரௌ': 'rau', 'ல': 'la', 'லா': 'lā', 'லி': 'li', 'லீ': 'lī', 'லு': 'lu', 'லூ': 'lū',
        'லெ': 'le', 'லே': 'lē', 'லை': 'lai', 'லொ': 'lo', 'லோ': 'lō', 'லௌ': 'lau', 'வ': 'va', 'வா': 'vā', 'வி': 'vi', 'வீ': 'vī', 'வு': 'vu',
        'வூ': 'vū', 'வெ': 've', 'வே': 'vē', 'வை': 'vai', 'வொ': 'vo', 'வோ': 'vō', 'வௌ': 'vau', 'ழ': 'lzha', 'ழா': 'lzhā', 'ழி': 'lzhi', 'ழீ': 'lzhī',
        'ழு': 'lzhu', 'ழூ': 'lzhū', 'ழெ': 'lzhe', 'ழே': 'lzhē', 'ழை': 'lzhai', 'ழொ': 'lzho', 'ழோ': 'lzhō', 'ழௌ': 'lzhau', 'ள': 'lla', 'ளா': 'llā',
        'ளி': 'lli', 'ளீ': 'llī', 'ளு': 'llu', 'ளூ': 'llū', 'ளெ': 'lle', 'ளே': 'llē', 'ளை': 'llai', 'ளொ': 'llo', 'ளோ': 'llō', 'ளௌ': 'llau',
        'ற': 'ṟa', 'றா': 'ṟā', 'றி': 'ṟi', 'றீ': 'ṟī', 'று': 'ṟu', 'றூ': 'ṟū', 'றெ': 'ṟe', 'றே': 'ṟē', 'றை': 'ṟai', 'றொ': 'ṟo', 'றோ': 'ṟō',
        'றௌ': 'ṟau', 'ன': 'ṉa', 'னா': 'ṉā', 'னி': 'ṉi', 'னீ': 'ṉī', 'னு': 'ṉu', 'னூ': 'ṉū', 'னெ': 'ṉe', 'னே': 'ṉē', 'னை': 'ṉai', 'னொ': 'ṉo',
        'னோ': 'ṉō', 'னௌ': 'ṉau'
    }

    rev_true_dict = {value: key for key, value in true_dict.items()}

    phon_dict = {'அ':'a','ஔ':'ā','ஈ':'i','ச்':'ī','ட்':'u','ண்':'ū','த்':'e','ந்':'ē','ப்':'ai','ய்':'o','ம்':'ō','ர்':'au','ல்':'ak','வ்':'k','ழ்':'ṅ',
        'ள்':'c','ற்':'ñ','ன்':'ṭ','ஆ':'ṇ','இ':'t','உ':'n','ஊ':'p','எ':'m','ஏ':'y','ஐ':'r','ஒ':'l','ஓ':'v','ஃ':'lzh','க்':'ll','ங்':'ṟ','ஜ்':'ṉ','Un':'f',
        'அ ':'அ',
        'ச':'ī','ட':'u','ண':'ū','த':'e','ந':'ē','ப':'ai','ய':'o','ம':'ō','ர':'au','ல':'ak','வ':'k','ழ':'ṅ',
        'ள':'c','ற':'ñ','ன':'ṭ','க':'ll','ங':'ṟ','ஞ':'ṉ','U':'f'
        }

    rev_phon = {value: key for key, value in phon_dict.items()}

    if len(hand_predictions) == 1:
        iss = str(phon_dict[hand_predictions[0][0]]).replace(" ","")
    else:
        iss = str(phon_dict[hand_predictions[1][0]]).replace(" ","")+str(phon_dict[hand_predictions[0][0]]).replace(" ","")
    
    if iss in rev_true_dict:
        return rev_true_dict[iss]
    else:
        return "unk"

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


true_dict = {
    'அ': 'a', 'ஆ': 'ā', 'இ': 'i', 'ஈ': 'ī', 'உ': 'u', 'ஊ': 'ū', 'எ': 'e', 'ஏ': 'ē', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'ō', 'ஔ': 'au', 'ஃ': 'ak',
    'க்': 'k', 'ங்': 'ṅ', 'ச்': 'c', 'ஞ்': 'ñ', 'ட்': 'ṭ', 'ண்': 'ṇ', 'த்': 't', 'ந்': 'n', 'ப்': 'p', 'ம்': 'm', 'ய்': 'y', 'ர்': 'r', 'ல்': 'l',
    'வ்': 'v', 'ழ்': 'lzh', 'ள்': 'll', 'ற்': 'ṟ', 'ன்': 'ṉ', 'க': 'ka', 'கா': 'kā', 'கி': 'ki', 'கீ': 'kī', 'கு': 'ku', 'கூ': 'kū', 'கெ': 'ke',
    'கே': 'kē', 'கை': 'kai', 'கொ': 'ko', 'கோ': 'kō', 'கௌ': 'kau', 'ங': 'nga', 'ஙா': 'ngā', 'ஙி': 'ngi', 'ஙீ': 'ngī', 'ஙு': 'ngu', 'ஙூ': 'ngū',
    'ஙெ': 'nge', 'ஙே': 'ngē', 'ஙை': 'ngai', 'ஙொ': 'ngo', 'ஙோ': 'ngō', 'ஙௌ': 'ngau', 'ச': 'sa', 'சா': 'sā', 'சி': 'si', 'சீ': 'sī', 'சு': 'su',
    'சூ': 'sū', 'செ': 'se', 'சே': 'sē', 'சை': 'sai', 'சொ': 'so', 'சோ': 'sō', 'சௌ': 'sau', 'ஞ': 'ña', 'ஞா': 'ñā', 'ஞி': 'ñi', 'ஞீ': 'ñī',
    'ஞு': 'ñu', 'ஞூ': 'ñū', 'ஞெ': 'ñe', 'ஞே': 'ñē', 'ஞை': 'ñai', 'ஞொ': 'ño', 'ஞோ': 'ñō', 'ஞௌ': 'ñau', 'ட': 'ṭa', 'டா': 'ṭā', 'டி': 'ṭi',
    'டீ': 'ṭī', 'டு': 'ṭu', 'டூ': 'ṭū', 'டெ': 'ṭe', 'டே': 'ṭē', 'டை': 'ṭai', 'டொ': 'ṭo', 'டோ': 'ṭō', 'டௌ': 'ṭau', 'ண': 'ṇa', 'ணா': 'ṇā',
    'ணி': 'ṇi', 'ணீ': 'ṇī', 'ணு': 'ṇu', 'ணூ': 'ṇū', 'ணெ': 'ṇe', 'ணே': 'ṇē', 'ணை': 'ṇai', 'ணொ': 'ṇo', 'ணோ': 'ṇō', 'ணௌ': 'ṇau', 'த': 'ta',
    'தா': 'tā', 'தி': 'ti', 'தீ': 'tī', 'து': 'tu', 'தூ': 'tū', 'தெ': 'te', 'தே': 'tē', 'தை': 'tai', 'தொ': 'to', 'தோ': 'tō', 'தௌ': 'tau',
    'ந': 'na', 'நா': 'nā', 'நி': 'ni', 'நீ': 'nī', 'நு': 'nu', 'நூ': 'nū', 'நெ': 'ne', 'நே': 'nē', 'நை': 'nai', 'நொ': 'no', 'நோ': 'nō',
    'நௌ': 'nau', 'ப': 'pa', 'பா': 'pā', 'பி': 'pi', 'பீ': 'pī', 'பு': 'pu', 'பூ': 'pū', 'பெ': 'pe', 'பே': 'pē', 'பை': 'pai', 'பொ': 'po',
    'போ': 'pō', 'பௌ': 'pau', 'ம': 'ma', 'மா': 'mā', 'மி': 'mi', 'மீ': 'mī', 'மு': 'mu', 'மூ': 'mū', 'மெ': 'me', 'மே': 'mē', 'மை': 'mai',
    'மொ': 'mo', 'மோ': 'mō', 'மௌ': 'mau', 'ய': 'ya', 'யா': 'yā', 'யி': 'yi', 'யீ': 'yī', 'யு': 'yu', 'யூ': 'yū', 'யெ': 'ye', 'யே': 'yē',
    'யை': 'yai', 'யொ': 'yo', 'யோ': 'yō', 'யௌ': 'yau', 'ர': 'ra', 'ரா': 'rā', 'ரி': 'ri', 'ரீ': 'rī', 'ரு': 'ru', 'ரூ': 'rū', 'ரெ': 're',
    'ரே': 'rē', 'ரை': 'rai', 'ரொ': 'ro', 'ரோ': 'rō', 'ரௌ': 'rau', 'ல': 'la', 'லா': 'lā', 'லி': 'li', 'லீ': 'lī', 'லு': 'lu', 'லூ': 'lū',
    'லெ': 'le', 'லே': 'lē', 'லை': 'lai', 'லொ': 'lo', 'லோ': 'lō', 'லௌ': 'lau', 'வ': 'va', 'வா': 'vā', 'வி': 'vi', 'வீ': 'vī', 'வு': 'vu',
    'வூ': 'vū', 'வெ': 've', 'வே': 'vē', 'வை': 'vai', 'வொ': 'vo', 'வோ': 'vō', 'வௌ': 'vau', 'ழ': 'lzha', 'ழா': 'lzhā', 'ழி': 'lzhi', 'ழீ': 'lzhī',
    'ழு': 'lzhu', 'ழூ': 'lzhū', 'ழெ': 'lzhe', 'ழே': 'lzhē', 'ழை': 'lzhai', 'ழொ': 'lzho', 'ழோ': 'lzhō', 'ழௌ': 'lzhau', 'ள': 'lla', 'ளா': 'llā',
    'ளி': 'lli', 'ளீ': 'llī', 'ளு': 'llu', 'ளூ': 'llū', 'ளெ': 'lle', 'ளே': 'llē', 'ளை': 'llai', 'ளொ': 'llo', 'ளோ': 'llō', 'ளௌ': 'llau',
    'ற': 'ṟa', 'றா': 'ṟā', 'றி': 'ṟi', 'றீ': 'ṟī', 'று': 'ṟu', 'றூ': 'ṟū', 'றெ': 'ṟe', 'றே': 'ṟē', 'றை': 'ṟai', 'றொ': 'ṟo', 'றோ': 'ṟō',
    'றௌ': 'ṟau', 'ன': 'ṉa', 'னா': 'ṉā', 'னி': 'ṉi', 'னீ': 'ṉī', 'னு': 'ṉu', 'னூ': 'ṉū', 'னெ': 'ṉe', 'னே': 'ṉē', 'னை': 'ṉai', 'னொ': 'ṉo',
    'னோ': 'ṉō', 'னௌ': 'ṉau'
}


rev_true_dict = {value: key for key, value in true_dict.items()}

labels_dict = {0: 'அ ', 1: 'ஆ ', 2: 'இ ', 3: 'ஈ ', 4: 'உ ', 
            5: 'ஊ ', 6: 'எ ', 7:'ஏ ', 8:'ஐ ',9:'ஒ ', 10:'ஓ ',
            11:'ஔ ', 12:'ஃ', 13:'க்', 14:'ங்', 15:'ச்', 16:'ஞ்',
            17:'ட்', 18:'ண்', 19:'த்', 20:'ந்', 21:'ப்', 22:'ம்',
            23:'ய்', 24:'ர்', 25:'ல்', 26:'வ்', 27:'ழ்', 28:'ள்', 
            29:'ற்', 30:'ன்'}

@app.route('/')
def index():
    template = "index.html"
    return render_template(template)

def generate_frames():
    global final_word
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    last_print_time = time.time()
    confidence_threshold = 0.5 
    space_count = 0
    word = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_predictions = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                for landmark in hand_landmarks.landmark:
                    normalized_x = landmark.x - wrist_x
                    normalized_y = landmark.y - wrist_y
                    normalized_z = landmark.z - wrist_z
                    data_aux.extend([normalized_x, normalized_y, normalized_z])

                if len(data_aux) == 63:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_index = int(prediction[0])
                    predicted_character = labels_dict.get(predicted_index , "Unknown")
                    confidence_scores = model.predict_proba([np.asarray(data_aux)])
                    confidence = np.max(confidence_scores)

                    if confidence < confidence_threshold:
                        predicted_character = "Unknown"

                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    label_text = f"{predicted_character} ({handedness.classification[0].label})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    hand_predictions.append(f"{predicted_character} ({handedness.classification[0].label})")

        current_time = time.time()
        if current_time - last_print_time >= 3:
            if hand_predictions:
                res = util(hand_predictions)
                if res != "unk":
                    print("Predicted Gestures: " + res)
                    word += res
                    last_print_time = time.time()
                else:
                    space_count += 1
                    if space_count >= 2:
                        with word_lock:  # Acquire the lock before modifying the global word
                            final_word = word
                        print(word)
                        text_to_speech(word)
                        space_count = 0
                        word = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_text')
def gesture_text():
    global final_word
    print("from gt",final_word)
    return jsonify({"predicted_word": final_word})


if __name__ == '__main__':
    app.run(debug=True)