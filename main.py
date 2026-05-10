import os
import gdown
import io
import random
import requests
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

MODEL_PATH = "dog_model_final.keras"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        url = "https://drive.google.com/uc?id=15s4lneWlkWg_Acf2NE5szuIZkBXnR3bl",
        MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")



class_labels = ["angry", "happy", "relaxed", "sad"]


emotion_tips = {
    "angry": [
        "يبدو أن كلبك يشعر بالتوتر أو الانزعاج في هذه اللحظة. يُنصح بعدم الاقتراب منه بشكل مفاجئ أو محاولة لمسه مباشرة. حاول منحه مساحة آمنة وهادئة ليهدأ تدريجيًا، وراقب البيئة المحيطة فقد يكون هناك صوت أو شيء يسبب له القلق.",

        "عندما يظهر الكلب علامات الغضب، فهذا قد يكون بسبب شعوره بالخطر أو التهديد. تجنب التفاعل المباشر معه في هذه الحالة، وابتعد عن أي سلوك قد يزيد من توتره. الأفضل هو تقليل الضوضاء ومنحه وقتًا كافيًا ليستعيد هدوءه.",

        "الغضب لدى الكلاب غالبًا يكون رد فعل على ضغط أو خوف. حاول تحديد السبب مثل وجود أشخاص غرباء أو أصوات مزعجة. امنح كلبك بيئة هادئة وآمنة، ولا تجبره على التفاعل حتى يهدأ تمامًا."
    ],

    "happy": [
        "يبدو أن كلبك في حالة مزاجية جيدة جدًا ويشعر بالسعادة والراحة. هذه فرصة مثالية للتفاعل الإيجابي معه مثل اللعب أو تقديم مكافأة بسيطة. استمرار هذه البيئة الإيجابية يساعد في تعزيز سلوكه الجيد.",

        "سعادة الكلب تعكس شعوره بالأمان والارتياح في محيطه. يمكنك استغلال هذه اللحظة لقضاء وقت ممتع معه أو تدريبه على أوامر جديدة بطريقة إيجابية. حافظ على نفس الأجواء الهادئة والمريحة.",

        "عندما يكون الكلب سعيدًا، يكون أكثر تقبلًا للتفاعل واللعب. حاول تعزيز هذه الحالة من خلال الاهتمام به، ومكافأته على سلوكه الجيد، وتوفير بيئة مليئة بالحب والأمان."
    ],

    "relaxed": [
        "يبدو أن كلبك في حالة هدوء واسترخاء، وهذا مؤشر جيد على شعوره بالأمان في المكان. من الأفضل الحفاظ على هذا الجو المريح وعدم إزعاجه، خاصة إذا كان مستلقيًا أو في وضعية راحة.",

        "الاسترخاء عند الكلاب يعني أنها تشعر بالثقة والراحة في محيطها. حاول الحفاظ على بيئة هادئة ومستقرة، وتجنب أي تغييرات مفاجئة قد تسبب له التوتر.",

        "حالة الهدوء هذه تعتبر مثالية لراحة الكلب. يمكنك تركه يسترخي دون تدخل، أو التفاعل معه بلطف شديد إذا أردت، مع الحفاظ على الهدوء العام في المكان."
    ],

    "sad": [
        "قد يبدو كلبك حزينًا أو منخفض النشاط، وهذا قد يكون بسبب الشعور بالوحدة أو التعب. حاول قضاء وقت هادئ معه وتقديم بعض الاهتمام دون إجباره على اللعب أو التفاعل.",

        "الحزن عند الكلاب قد يظهر على شكل خمول أو قلة حركة. تأكد من أن كلبك يتناول طعامه بشكل طبيعي، وأنه لا يعاني من أي مشكلة صحية. الاهتمام الهادئ يمكن أن يساعد في تحسين حالته.",

        "إذا استمرت علامات الحزن لفترة طويلة، فمن الأفضل مراقبة سلوك الكلب بشكل دقيق. قد يكون بحاجة إلى اهتمام أكثر أو حتى استشارة طبيب بيطري إذا لاحظت تغيرات مستمرة."
    ]

    
}

def get_random_tip(emotion):
    return random.choice(
        emotion_tips.get(emotion, ["راقب حالة الحيوان وتأكد من راحته وسلامته."])
    )

def get_one_youtube_video(emotion, animal="dog"):
    if not YOUTUBE_API_KEY:
        return None

    query_map = {
        "angry": "dog aggressive behavior signs veterinarian",
        "happy": "dog happy body language signs",
        "relaxed": "dog relaxed body language signs",
        "sad": "dog sad behavior signs veterinarian"
    }

    query = query_map.get(emotion, f"{animal} behavior signs")

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 1,
        "key": YOUTUBE_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "items" not in data or len(data["items"]) == 0:
        return None

    item = data["items"][0]
    video_id = item["id"]["videoId"]

    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
    }

def preprocess_uploaded_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.get("/")
def home():
    return {"message": "Dog Emotion Detection API is running"}

@app.post("/analyze-dog")
async def analyze_dog(file: UploadFile = File(...)):
    file_bytes = await file.read()

    img_array = preprocess_uploaded_image(file_bytes)

    prediction = model.predict(img_array)[0]

    predicted_index = int(np.argmax(prediction))
    emotion = class_labels[predicted_index]
    confidence = float(prediction[predicted_index])

    probabilities = {
        class_labels[i]: float(prediction[i])
        for i in range(len(class_labels))
    }

    return {
        "emotion": emotion,
        "confidence": confidence,
        "tip": get_random_tip(emotion),
        "video": get_one_youtube_video(emotion, "dog"),
        "probabilities": probabilities
    }
