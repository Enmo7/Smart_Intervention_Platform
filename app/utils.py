# model_inference.py
import os
from transformers import pipeline

# المسار إلى المودل المدرب الذي قمنا بحفظه
MODEL_PATH = "./my_arabic_risk_model"

# تأكد من أن المودل تم تدريبه وحفظه قبل محاولة تحميله
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run model_training.py first.")

# تحميل الـ pipeline مرة واحدة فقط عند بدء تشغيل الخادم لتحسين الأداء
classifier = pipeline("text-classification", model=MODEL_PATH)

def classify_text_for_risk(text: str):
    """
    يصنف النص المدخل إلى مستوى خطر (منخفض, متوسط, عالي) باستخدام المودل المدرب.
    """
    try:
        prediction = classifier(text)[0]
        # تحويل "LABEL_0" إلى 0، "LABEL_1" إلى 1، وهكذا
        label_id = int(prediction['label'].replace('LABEL_', ''))
        return {
            "label_id": label_id, # 0: منخفض، 1: متوسط، 2: عالي
            "label_name": prediction['label'],
            "confidence": prediction['score']
        }
    except Exception as e:
        print(f"Error during text classification: {e}")
        return {
            "label_id": -1, # قيمة تشير إلى خطأ
            "label_name": "error",
            "confidence": 0.0
        }

if __name__ == "__main__":
    # مثال على كيفية الاستخدام
    test_text_low = "العب بشكل طبيعي وأستمتع"
    test_text_medium = "أحتاج أن أنام لكن لا أستطيع بسبب اللعبة"
    test_text_high = "أفكر في إيذاء نفسي"

    print(f"'{test_text_low}': {classify_text_for_risk(test_text_low)}")
    print(f"'{test_text_medium}': {classify_text_for_risk(test_text_medium)}")
    print(f"'{test_text_high}': {classify_text_for_risk(test_text_high)}")