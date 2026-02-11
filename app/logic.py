def get_intervention(label_name):
    """
    Mapping AI sentiment labels to risk interventions.
    Matching labels from your BERT model: Positive, Negative, Neutral.
    """
    database = {
        "Negative": {
            "title": "تنبيه: مؤشرات خطورة عالية",
            "message": "نبرة الحديث تشير إلى حالة نفسية سلبية أو ضيق شديد. نحن نهتم لأمرك، يرجى مراجعة مختص للتحدث.",
            "video": "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "color": "#ff4d4d"
        },
        "Neutral": {
            "title": "تنبيه: حالة غير مستقرة",
            "message": "النبرة تبدو محايدة ولكنها قد تحمل مؤشرات إجهاد رقمي. خذ قسطاً من الراحة بعيداً عن الشاشة.",
            "video": "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "color": "#ffa500"
        },
        "Positive": {
            "title": "حالة آمنة وإيجابية",
            "message": "المشاعر إيجابية ومستقرة. استمتع بوقتك في اللعب وحافظ على هذا التوازن الجميل!",
            "video": None,
            "color": "#00ff88"
        }
    }
    # Default to Neutral if label is unknown
    return database.get(label_name, database["Neutral"])