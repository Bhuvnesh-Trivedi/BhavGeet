import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
header_html = """
<div style="background-color:#1a1a1a; padding:20px; border-radius:10px; text-align:center;">
    <h1 style="color:#f0f0f0; font-family:'Arial', sans-serif; margin-bottom:0;">BHAVGEET</h1>
    <p style="color:#b0b0b0; font-size:18px; margin-top:5px;"><ul>Emotion Based Music Recommender</ul></p>
</div>
"""


st.markdown(header_html, unsafe_allow_html=True)


if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


	

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"




footer_html = """
<div style="background-color:#1a1a1a; padding:15px; border-radius:10px; text-align:center; margin-top:20px;">
    <p style="color:#b0b0b0; font-size:16px; margin:0;">
        © 2024 BHAVGEET | 
        <a href="https://linkedin.com/in/bhuvnesh-trivedi-50a28623a" style="color:#f0f0f0; text-decoration:none;">LinkedIn</a> |
        <a href="https://github.com/Bhuvnesh-Trivedi" style="color:#f0f0f0; text-decoration:none;">GitHub</a> |
        <a href="https://x.com/Bhuvi_____" style="color:#f0f0f0; text-decoration:none;">Twitter</a>
    </p>
    <p style="color:#707070; font-size:14px; margin-top:5px;">Designed with ❤️ in Python</p>
</div>
"""


st.markdown(footer_html, unsafe_allow_html=True)
