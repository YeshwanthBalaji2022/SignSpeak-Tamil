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


DEFAULT_SAMPLING_RATE = 16000

def text_to_speech(text, lang='ta'):
    if not text.strip():
        # Do nothing if the text input is empty or blank
        return
    
    try:
        # Generate speech from text using gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save the speech to a BytesIO object
        byte_io = io.BytesIO()
        tts.write_to_fp(byte_io)
        byte_io.seek(0)
        
        # Initialize pygame mixer with the appropriate sampling rate
        pygame.mixer.init(frequency=DEFAULT_SAMPLING_RATE)
        
        # Load the speech data into pygame
        pygame.mixer.music.load(byte_io, 'mp3')
        pygame.mixer.music.play()
        
        # Wait until the speech has finished playing
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
# label_encoder = model_dict['label_encoder']
# Define the gesture names

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

# print(rev_phon['a'])

# enc = {'அ':'அ','ஔ':'ஆ','ஈ':'இ','ச்':'ஈ','ட்':'உ','ண்':'ஊ','த்':'எ','ந்':'ஏ','ப்':'ஐ','ய்':'ஒ','ம்':'ஓ','ர்':'ஔ','ல்':'ஃ','வ்':'க்','ழ்':'ங்',
#        'ள்':'ச்','ற்':'ஞ்','ன்':'ட்','ஆ':'ண்','இ':'த்','உ':'ந்','ஊ':'ப்','எ':'ம்','ஏ':'ய்','ஐ':'ர்','ஒ':'ல்','ஓ':'வ்','ஃ':'ழ்','க்':'ள்','ங்':'ற்','ஜ்':'ன்','Un':'fk',
#        'அ ':'அ',
#        'ச':'ஈ','ட':'உ','ண':'ஊ','த':'எ','ந':'ஏ','ப':'ஐ','ய':'ஒ','ம':'ஓ','ர':'ஔ','ல':'ஃ','வ':'க்','ழ':'ங்',
#        'ள':'ச்','ற':'ஞ்','ன':'ட்','க':'ள்','ங':'ற்','ஞ':'ன்','U':'fk'
#        }
# labels_dict = {0: 'a', 1: 'k', 2: 'r', 3: 'v', 4: 'lzh', 
#             5: 'la', 6: 'ra', 7:'ann', 8:'<unk>',9:'aa, single o', 10:'aa',
#             11:'i', 12:'ii', 13:'u', 14:'e', 15:'ee', 16:'ai',
#             17:'oo', 18:'au', 19:'ak', 20:'n', 21:'c', 22:'nn',
#             23:'t', 24:'na', 25:'ta', 26:'an', 27:'p', 28:'m', 
#             29:'y', 30:'l'}
# shift by "something" consistently even after remapping
# labels_dict = {0: 'a', 1: 'aa', 2: 'i', 3: 'ii', 4: 'u'}

# Initialize webcam and Mediapipe Hands solution
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Initialize the last print time
last_print_time = time.time()

# Define a confidence threshold
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

    hand_predictions = []  # List to store predictions for both hands

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Prepare data for prediction
            data_aux = []
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

            for landmark in hand_landmarks.landmark:
                normalized_x = landmark.x - wrist_x
                normalized_y = landmark.y - wrist_y
                normalized_z = landmark.z - wrist_z

                data_aux.extend([normalized_x, normalized_y, normalized_z])

            if len(data_aux) == 63:  # Ensure the data length is correct for each hand
                prediction = model.predict([np.asarray(data_aux)])
                predicted_index = int(prediction[0])
                # print(prediction)
                predicted_character = labels_dict.get(predicted_index , "Unknown")
                # predicted_character = label_encoder.inverse_transform([predicted_index])[0]
                # Calculate confidence score (assuming the model has a method for this, e.g., predict_proba)
                confidence_scores = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(confidence_scores)

                if confidence < confidence_threshold:
                    predicted_character = "Unknown"

                # Get bounding box coordinates
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Display prediction on the frame
                label_text = f"{predicted_character} ({handedness.classification[0].label})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # Store the prediction
                hand_predictions.append(f"{predicted_character} ({handedness.classification[0].label})")

    # Print the predictions in the console every 0.5 seconds
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
                    print(word)
                    text_to_speech(word)
                    space_count = 0
                    word = ""
            # print("Predicted Gestures: " + ", ".join(util(hand_predictions)))
            # enc[hand_predictions[0][0]]
            # str(enc[hand_predictions[0][0]])

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

##############################################################
# import warnings
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Ignore specific warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# # Load the trained model and label encoder
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
# label_encoder = model_dict['label_encoder']

# # Initialize webcam and Mediapipe Hands solution
# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# # Initialize the last print time
# last_print_time = time.time()

# # Define a confidence threshold
# confidence_threshold = 0.5

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     hand_predictions = []  # List to store predictions for both hands

#     if results.multi_hand_landmarks and results.multi_handedness:
#         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#             # Draw hand landmarks on the frame
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#             # Prepare data for prediction
#             data_aux = []
#             wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
#             wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
#             wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

#             for landmark in hand_landmarks.landmark:
#                 normalized_x = landmark.x - wrist_x
#                 normalized_y = landmark.y - wrist_y
#                 normalized_z = landmark.z - wrist_z

#                 data_aux.extend([normalized_x, normalized_y, normalized_z])

#             if len(data_aux) == 63:  # Ensure the data length is correct for each hand
#                 prediction = model.predict([np.asarray(data_aux)])
#                 predicted_index = int(prediction[0])
#                 print(prediction)
#                 predicted_character = label_encoder.inverse_transform([predicted_index])[0]

#                 # Calculate confidence score (assuming the model has a method for this, e.g., predict_proba)
#                 confidence_scores = model.predict_proba([np.asarray(data_aux)])
#                 confidence = np.max(confidence_scores)

#                 if confidence < confidence_threshold:
#                     predicted_character = "Unknown"

#                 # Get bounding box coordinates
#                 x_ = [landmark.x for landmark in hand_landmarks.landmark]
#                 y_ = [landmark.y for landmark in hand_landmarks.landmark]
#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10
#                 x2 = int(max(x_) * W) + 10
#                 y2 = int(max(y_) * H) + 10

#                 # Display prediction on the frame
#                 label_text = f"{predicted_character} ({handedness.classification[0].label})"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#                 # Store the prediction
#                 hand_predictions.append(f"{predicted_character} ({handedness.classification[0].label})")

#     # Print the predictions in the console every 0.5 seconds
#     current_time = time.time()
#     if current_time - last_print_time >= 0.5:
#         if hand_predictions:
#             print("Predicted Gestures: " + ", ".join(hand_predictions))
#         last_print_time = current_time

#     # Display the frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Ignore specific warnings
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# # Load the trained models and label encoders for left and right hands
# left_hand_model_dict = pickle.load(open('left_hand_model.p', 'rb'))
# left_hand_model = left_hand_model_dict['model']
# left_hand_label_encoder = left_hand_model_dict['label_encoder']

# right_hand_model_dict = pickle.load(open('right_hand_model.p', 'rb'))
# right_hand_model = right_hand_model_dict['model']
# right_hand_label_encoder = right_hand_model_dict['label_encoder']

# # Initialize webcam and Mediapipe Hands solution
# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# # Initialize the last print time
# last_print_time = time.time()

# # Define a confidence threshold
# confidence_threshold = 0.3

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     hand_predictions = []  # List to store predictions for both hands

#     if results.multi_hand_landmarks and results.multi_handedness:
#         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#             # Draw hand landmarks on the frame
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#             # Prepare data for prediction
#             data_aux = []
#             wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
#             wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
#             wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

#             for landmark in hand_landmarks.landmark:
#                 normalized_x = landmark.x - wrist_x
#                 normalized_y = landmark.y - wrist_y
#                 normalized_z = landmark.z - wrist_z

#                 data_aux.extend([normalized_x, normalized_y, normalized_z])

#             if len(data_aux) == 63:  # Ensure the data length is correct for each hand
#                 data_aux = np.asarray(data_aux).reshape(1, -1)
                
#                 if handedness.classification[0].label == 'Left':
#                     model = left_hand_model
#                     label_encoder = left_hand_label_encoder
#                 else:
#                     model = right_hand_model
#                     label_encoder = right_hand_label_encoder
                
#                 prediction = model.predict(data_aux)
#                 predicted_index = int(prediction[0])
#                 predicted_character = label_encoder.inverse_transform([predicted_index])[0]

#                 # Calculate confidence score (optional, based on model output)
#                 confidence_scores = model.predict_proba(data_aux)
#                 confidence = np.max(confidence_scores)

#                 if confidence < confidence_threshold:
#                     predicted_character = "Unknown"

#                 # Get bounding box coordinates
#                 x_ = [landmark.x for landmark in hand_landmarks.landmark]
#                 y_ = [landmark.y for landmark in hand_landmarks.landmark]
#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10
#                 x2 = int(max(x_) * W) + 10
#                 y2 = int(max(y_) * H) + 10

#                 # Display prediction on the frame
#                 label_text = f"{predicted_character} ({handedness.classification[0].label})"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#                 # Store the prediction
#                 hand_predictions.append(f"{predicted_character} ({handedness.classification[0].label})")

#     # Print the predictions in the console every 0.5 seconds
#     current_time = time.time()
#     if current_time - last_print_time >= 0.5:
#         if hand_predictions:
#             print("Predicted Gestures: " + ", ".join(hand_predictions))
#         last_print_time = current_time

#     # Display the frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
