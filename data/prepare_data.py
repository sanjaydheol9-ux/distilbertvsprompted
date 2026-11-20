#!/usr/bin/env python3
"""
prepare_data.py
Downloads, cleans, formats, flags code-mixed sentences, and stratifies 80/20 split into data/train.csv and data/test.csv.
Supports --mock flag to generate high-quality realistic datasets locally without Hugging Face / GPU dependencies.
"""

import os
import re
import sys
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Regex patterns for Indic scripts and Latin words
DEV_REGEX = re.compile(r'[\u0900-\u097F]')  # Devanagari (Hindi)
KAN_REGEX = re.compile(r'[\u0C80-\u0CFF]')  # Kannada
TEL_REGEX = re.compile(r'[\u0C00-\u0C7F]')  # Telugu
LATIN_REGEX = re.compile(r'[a-zA-Z]{2,}')   # Latin script tokens (English / transliteration)

def is_code_mixed(text: str) -> bool:
    """
    Heuristic rule to detect code-mixing:
    Checks if text contains both native Indic script characters and Latin characters (words with >= 2 letters),
    or Latin-script transliterations with Indic slang/keywords.
    """
    if not isinstance(text, str):
        return False
    has_dev = bool(DEV_REGEX.search(text))
    has_kan = bool(KAN_REGEX.search(text))
    has_tel = bool(TEL_REGEX.search(text))
    has_indic = has_dev or has_kan or has_tel
    has_latin = bool(LATIN_REGEX.search(text))
    
    # If it has both Indic script and Latin words -> definitely code-mixed
    if has_indic and has_latin:
        return True
    
    # Also detect Romanized/transliterated code-mixing (Latin script containing typical Indian language particles)
    translit_tokens = {
        'hai', 'tha', 'nahi', 'bahut', 'achha', 'accha', 'bekar', 'mast', 'aur', 'kya',
        'chala', 'bagundi', 'ledu', 'kadu', 'super', 'undhi', 'chesanu', 'inka',
        'chennagide', 'beku', 'illa', 'thumba', 'tumba', 'sikkapatte', 'kelasa', 'agalla'
    }
    tokens = set(re.findall(r'[a-zA-Z]+', text.lower()))
    if len(tokens.intersection(translit_tokens)) >= 2:
        return True

    return False

# Sample bank of authentic phrases for realistic mock generation
MOCK_CORPUS = {
    "hi": {
        "positive": [
            "यह फिल्म सच में बहुत शानदार और दिल को छू लेने वाली है।",
            "उत्पाद की गुणवत्ता बेहतरीन है, समय पर डिलीवरी मिली।",
            "कहानी बहुत प्रेरणादायक है और सभी कलाकारों का अभिनय लाजवाब है।",
            "पैसा वसूल अनुभव रहा, मैं इसे सभी को देखने की सलाह दूंगा।",
            "कैमरा और बैटरी बैकअप उम्मीद से कहीं बेहतर निकला।",
            "खाना बहुत स्वादिष्ट था और सेवा भी अत्यंत विनम्र थी।",
            "यह फोन इस बजट में सबसे अच्छा विकल्प साबित हुआ है।",
            "म्यूजिक और बैकग्राउंड स्कोर ने फिल्म में जान डाल दी।",
            "पैकेजिंग बहुत सुरक्षित थी और सामान बिल्कुल सही हालत में मिला।",
            "इस किताब ने मेरी सोच बदल दी, हर छात्र को पढ़ना चाहिए।"
        ],
        "negative": [
            "बिलकुल बेकार उत्पाद है, दो दिन में ही खराब हो गया।",
            "फिल्म की कहानी बेहद उबाऊ और निराशाजनक थी।",
            "ग्राहक सेवा का रवैया बहुत खराब और असहयोगपूर्ण रहा।",
            "पैसे की पूरी बर्बादी, कभी भी इस ब्रांड पर भरोसा न करें।",
            "डिलीवरी में एक हफ्ते की देरी हुई और डिब्बा भी टूटा हुआ था।",
            "साउंड क्वालिटी बहुत घटिया है, बेस बिल्कुल भी नहीं है।",
            "निर्देशन बहुत कमजोर था और क्लाइमेक्स बिना किसी तर्क के खत्म हुआ।",
            "तस्वीर में कुछ और दिखाया गया था और असलियत में कुछ और निकला।",
            "स्क्रीन बहुत जल्दी गर्म हो जाती है, बैटरी एक घंटा भी नहीं टिकती।",
            "बेहद घटिया अनुभव, मैं इसे किसी को भी लेने की सलाह नहीं दूंगा।"
        ],
        "code_mixed_pos": [
            "Yeh movie sach me bahut awesome thi, acting super hit hai!",
            "Product quality is outstanding, packaging bhi ekdum top notch thi.",
            "Display aur battery performance kamaal ka hai, paisa vasool deal!",
            "Story thodi slow thi but climax emotional and heart-touching laga.",
            "Service was fast, staff ka behavior bahut polite and cooperative tha."
        ],
        "code_mixed_neg": [
            "Very bad experience yaar, product do din me dead ho gaya.",
            "Screen quality totally bakwas hai, full brightness me bhi dikhta nahi.",
            "Customer support useless hai, call disconnect kar dete hain.",
            "Overhyped film hai, story me zero logic aur direction pathetic tha.",
            "Delivery bahut delay hui aur parcel damaged condition me deliver hua."
        ]
    },
    "kn": {
        "positive": [
            "ಈ ಚಲನಚಿತ್ರ ನಿಜವಾಗಿಯೂ ಅದ್ಭುತವಾಗಿದೆ ಮತ್ತು ನಟನೆ ಅತ್ಯುತ್ತಮವಾಗಿದೆ.",
            "ಉತ್ಪನ್ನದ ಗುಣಮಟ್ಟ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, ನಿರೀಕ್ಷೆಗಿಂತ ಉತ್ತಮವಾಗಿದೆ.",
            "ಕಥೆ ತುಂಬಾ ಸ್ಪೂರ್ತಿದಾಯಕವಾಗಿದೆ ಮತ್ತು ಸಂಗೀತ ಮನಸ್ಸಿಗೆ ಮುದ ನೀಡುತ್ತದೆ.",
            "ಬೆಲೆಗೆ ತಕ್ಕಂತೆ ಅತ್ಯುತ್ತಮ ಮೌಲ್ಯವನ್ನು ನೀಡುವ ಸ್ಮಾರ್ಟ್‌ಫೋನ್.",
            "ಕ್ಯಾಮೆರಾ ಗುಣಮಟ್ಟ ಸೂಪರ್ ಆಗಿದೆ, ಬ್ಯಾಟರಿ ಬಾಳಿಕೆ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
            "ಆಹಾರ ರುಚಿಕರವಾಗಿತ್ತು ಮತ್ತು ಸೇವೆ ಅತ್ಯಂತ ವಿನಮ್ರವಾಗಿತ್ತು.",
            "ಪ್ಯಾಕಿಂಗ್ ತುಂಬಾ ಸುರಕ್ಷಿತವಾಗಿತ್ತು, ಸರಿಯಾದ ಸಮಯಕ್ಕೆ ತಲುಪಿದೆ.",
            "ಈ ಪುಸ್ತಕ ಪ್ರತಿಯೊಬ್ಬ ವಿದ್ಯಾರ್ಥಿಯೂ ಓದಲೇಬೇಕಾದ ಕೃತಿ.",
            "ನಟರ ಅಭಿನಯ ಮತ್ತು ನಿರ್ದೇಶನ ಎರಡು ಕೂಡ ಅದ್ಭುತವಾಗಿದೆ.",
            "ನಾನು ಇದನ್ನು ಎಲ್ಲರಿಗೂ ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ, ಅತ್ಯುತ್ತಮ ಅನುಭವ."
        ],
        "negative": [
            "ಸಂಪೂರ್ಣವಾಗಿ ವ್ಯರ್ಥ ಉತ್ಪನ್ನ, ಎರಡು ದಿನದಲ್ಲೇ ಹಾಳಾಗಿದೆ.",
            "ಸಿನಿಮಾ ಕಥೆ ತುಂಬಾ ನೀರಸವಾಗಿದೆ, ಸಮಯ ವ್ಯರ್ಥ.",
            "ಗ್ರಾಹಕ ಸೇವೆ ಅತ್ಯಂತ ಕಳಪೆಯಾಗಿದೆ, ಯಾವುದೇ ಸ್ಪಂದನೆ ಇಲ್ಲ.",
            "ಗುಣಮಟ್ಟ ತೀರಾ ಕೆಟ್ಟದಾಗಿದೆ, ಹಣ ವ್ಯರ್ಥವಾಯಿತು.",
            "ಡೆಲಿವರಿ ತುಂಬಾ ತಡವಾಯಿತು ಮತ್ತು ಪ್ಯಾಕೇಜ್ ಹಾಳಾಗಿತ್ತು.",
            "ಧ್ವನಿ ಗುಣಮಟ್ಟ ತುಂಬಾ ಕೆಟ್ಟದಾಗಿದೆ, ಸ್ಪಷ್ಟತೆ ಇಲ್ಲ.",
            "ಸ್ಕ್ರೀನ್ ಬೇಗ ಬಿಸಿಯಾಗುತ್ತದೆ, ಬ್ಯಾಟರಿ ಒಂದು ಗಂಟೆಯೂ ಬರುವುದಿಲ್ಲ.",
            "ಚಿತ್ರದಲ್ಲಿ ತೋರಿಸಿದ್ದು ಒಂದು, ನಮಗೆ ಬಂದಿದ್ದು ಇನ್ನೊಂದು.",
            "ನಿರ್ದೇಶನ ತುಂಬಾ ದುರ್ಬಲವಾಗಿದೆ, ಯಾವುದೇ ಹೊಸತನವಿಲ್ಲ.",
            "ಈ ಬ್ರ್ಯಾಂಡ್‌ ಅನ್ನು ಯಾರೂ ನಂಬಬೇಡಿ, ತೀರಾ ಕಳಪೆ ಅನುಭವ."
        ],
        "code_mixed_pos": [
            "Cinema tumba chennagide, BGM and visual effects super thrilling!",
            "Product quality awesome ide, timely delivery aaytu thumba happy.",
            "Screen clarity super agide, worth the price and great battery life.",
            "Story touching aagide, all actors performance really nice.",
            "Service super fast, packaging kuda safe aagi bandide."
        ],
        "code_mixed_neg": [
            "Tumba worst product, within one week completely damage aaytu.",
            "Movie story boring ide, first half slow and totally time waste.",
            "Customer support worst response, refund kuda innu aagilla.",
            "Sound clarity illa, heavy distortion and low bass.",
            "Battery backup tumba poor, within 2 hours charge khali aaguthe."
        ]
    },
    "te": {
        "positive": [
            "ఈ సినిమా చాలా అద్భుతంగా ఉంది, నటీనటుల నటన సూపర్.",
            "ఉత్పత్తి నాణ్యత చాలా బాగుంది, సమయానికి డెలివరీ అయింది.",
            "కథ ఎంతో స్ఫూర్తిదాయకంగా ఉంది మరియు సంగీతం మనసుకు హత్తుకుంది.",
            "ఈ ధరలో ఇది బెస్ట్ మొబైల్, కెమెరా మరియు బ్యాటరీ అద్భుతం.",
            "డిజైన్ చాలా ప్రీమియంగా ఉంది, స్క్రీన్ క్లారిటీ బాగుంది.",
            "ఆహారం చాలా రుచికరంగా ఉంది మరియు సర్వీస్ చాలా వేగంగా ఉంది.",
            "ప్యాకేజింగ్ చాలా పటిష్టంగా ఉంది, ఎటువంటి లోపం లేదు.",
            "దర్శకుడి ప్రతిభ ప్రతి సన్నివేశంలో స్పష్టంగా కనిపిస్తుంది.",
            "డబ్బులకు తగిన పూర్తి న్యాయం జరిగింది, అందరికీ సిఫార్సు చేస్తాను.",
            "చాలా మంచి అనుభవం, నేను చాలా సంతృప్తిగా ఉన్నాను."
        ],
        "negative": [
            "పూర్తిగా పనికిరాని వస్తువు, రెండు రోజులకే పాడైపోయింది.",
            "సినిమా కథ చాలా బోరింగ్‌గా ఉంది, సమయం వృధా.",
            "కస్టమర్ సర్వీస్ చాలా అధ్వానంగా ఉంది, ఎవరూ స్పందించడం లేదు.",
            "నాణ్యత చాలా తక్కువగా ఉంది, డబ్బులు వృధా అయ్యాయి.",
            "డెలివరీ చాలా ఆలస్యంగా జరిగింది, బాక్స్ కూడా పాడైంది.",
            "సౌండ్ క్వాలిటీ అస్సలు బాలేదు, చాలా నాయిస్ వస్తోంది.",
            "ఫోన్ చాలా వేడెక్కుతోంది, బ్యాటరీ అసలు నిలవడం లేదు.",
            "ఫోటోలలో చూసింది ఒకటి, డెలివరీ అయింది వేరొకటి.",
            "దర్శకత్వం చాలా బలహీనంగా ఉంది, ఎటువంటి లాజిక్ లేదు.",
            "ఈ బ్రాండ్‌ను ఎవరూ కొనవద్దు, చాలా చెత్త అనుభవం."
        ],
        "code_mixed_pos": [
            "Movie chala bagundi, hero performance and BGM next level!",
            "Product build quality awesome, delivery kuda super fast ga vachindi.",
            "Camera and display outstanding, full paisa vasool product.",
            "Family tho chudataniki perfect cinema, emotional scenes touching.",
            "Very good battery backup, continuous usage lo kuda 10 hours vastundi."
        ],
        "code_mixed_neg": [
            "Chala worst product bro, within 3 days lo work avvadam ledu.",
            "First half ok kani second half full boring and drag chesaru.",
            "Customer care service useless, call chesina no proper response.",
            "Sound quality zero, high volume lo speaker sound crack avtundi.",
            "Phone heavy heat avtundi, heating issue tho return pettanu."
        ]
    }
}

def generate_mock_dataset(total_train=10000, total_test=2500, seed=42):
    """
    Generates realistic, balanced datasets across Hindi, Kannada, Telugu
    with realistic label distributions and code-mixed reviews.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    languages = ["hi", "kn", "te"]
    labels = ["positive", "negative"]
    
    total_samples = total_train + total_test
    samples_per_lang = total_samples // len(languages)
    
    records = []
    
    modifiers_pos = [
        "", " నిజంగా!", " सचमुच बहुत बढ़िया।", " ಖಂಡಿತವಾಗಿಯೂ ಅದ್ಭುತ.",
        " Overall great experience.", " Must buy for everyone.", " 10/10 rating!",
        " Value for money.", " Super satisfied.", " Worth every penny."
    ]
    modifiers_neg = [
        "", " ఎప్పటికీ కొనవద్దు.", " बिल्कुल भी मत खरीदना।", " ದಯವಿಟ್ಟು ಯಾರೂ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.",
        " Total waste of money.", " Highly disappointed.", " 1 star rating.",
        " Regret buying this.", " Worst purchase ever.", " Horrible quality."
    ]
    
    for lang in languages:
        for _ in range(samples_per_lang):
            label = random.choice(labels)
            # 20-25% chance of code-mixed text
            is_cm_target = (random.random() < 0.22)
            
            if is_cm_target:
                pool = (MOCK_CORPUS[lang]["code_mixed_pos"] if label == "positive"
                        else MOCK_CORPUS[lang]["code_mixed_neg"])
                base_text = random.choice(pool)
                mod = random.choice(modifiers_pos if label == "positive" else modifiers_neg)
                text = f"{base_text}{mod}".strip()
            else:
                pool = (MOCK_CORPUS[lang]["positive"] if label == "positive"
                        else MOCK_CORPUS[lang]["negative"])
                base_text = random.choice(pool)
                # optionally add variations
                if random.random() < 0.4:
                    extra = random.choice(pool)
                    text = f"{base_text} {extra}".strip()
                else:
                    text = base_text
            
            cm_flag = is_code_mixed(text)
            records.append({
                "text": text,
                "label": label,
                "language": lang,
                "is_code_mixed": cm_flag
            })
            
    df = pd.DataFrame(records)
    # Stratified 80/20 split based on combined (language, label)
    df["strata"] = df["language"] + "_" + df["label"]
    train_df, test_df = train_test_split(
        df,
        test_size=total_test / total_samples,
        stratify=df["strata"],
        random_state=seed
    )
    
    train_df = train_df.drop(columns=["strata"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strata"]).reset_index(drop=True)
    
    return train_df, test_df

def download_and_prepare_real(data_dir: str, seed: int = 42):
    """
    Attempts to download IndicSentiment from HuggingFace, clean, map to binary labels,
    flag code-mixed, and perform stratified 80/20 train/test split.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[-] datasets package not installed. Run `pip install datasets` or use `--mock`.")
        sys.exit(1)
        
    print("[+] Attempting to download IndicSentiment from Hugging Face...")
    languages = ["hi", "kn", "te"]
    dfs = []
    
    for lang in languages:
        try:
            print(f"    Loading language: {lang}...")
            # Try primary dataset ai4bharat/IndicSentiment
            ds = load_dataset("ai4bharat/IndicSentiment", lang, split="train")
            sub_df = pd.DataFrame(ds)
            # Ensure standard column names
            if "indic_text" in sub_df.columns:
                sub_df["text"] = sub_df["indic_text"]
            elif "sentence" in sub_df.columns:
                sub_df["text"] = sub_df["sentence"]
            sub_df["language"] = lang
            dfs.append(sub_df)
        except Exception as e:
            print(f"    [!] Failed to load {lang} from ai4bharat/IndicSentiment: {e}")
            print(f"    [!] Trying fallback IndicXNLI or public dataset...")
            try:
                ds = load_dataset("Divyanshu/indic_sentiment_extracted", lang, split="train")
                sub_df = pd.DataFrame(ds)
                sub_df["language"] = lang
                dfs.append(sub_df)
            except Exception as e2:
                print(f"    [!] Fallback failed for {lang}: {e2}")
                
    if not dfs:
        print("[-] Could not download real dataset. Falling back to mock data generator.")
        return generate_mock_dataset()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Map labels to binary (positive / negative)
    # Filter out neutral or map 3-class/star ratings to binary
    label_map = {
        "Positive": "positive", "positive": "positive", "pos": "positive", 1: "positive", "1": "positive",
        "Negative": "negative", "negative": "negative", "neg": "negative", 0: "negative", "0": "negative",
        # 5-star rating mappings if applicable:
        4: "positive", 5: "positive", "4": "positive", "5": "positive",
        1: "negative", 2: "negative", "1": "negative", "2": "negative"
    }
    
    full_df["label"] = full_df["label"].map(label_map)
    full_df = full_df.dropna(subset=["label", "text"]).reset_index(drop=True)
    
    # Code-mixed flag
    full_df["is_code_mixed"] = full_df["text"].apply(is_code_mixed)
    
    # Stratified 80/20 split
    full_df["strata"] = full_df["language"] + "_" + full_df["label"]
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.20,
        stratify=full_df["strata"],
        random_state=seed
    )
    train_df = train_df.drop(columns=["strata"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strata"]).reset_index(drop=True)
    
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Prepare train and test data for Indic Sentiment Analysis.")
    parser.add_argument("--mock", action="store_true", help="Generate realistic mock dataset locally without HF download.")
    parser.add_argument("--train-size", type=int, default=10000, help="Train set size for mock generator (default: 10000).")
    parser.add_argument("--test-size", type=int, default=2500, help="Test set size for mock generator (default: 2500).")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory to save CSVs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    
    if args.mock:
        print(f"[+] Generating mock dataset (train: {args.train_size}, test: {args.test_size}, seed: {args.seed})...")
        train_df, test_df = generate_mock_dataset(total_train=args.train_size, total_test=args.test_size, seed=args.seed)
    else:
        print("[+] Preparing dataset from source...")
        train_df, test_df = download_and_prepare_real(args.output_dir, seed=args.seed)
        
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"[✓] Successfully saved:")
    print(f"    - {train_path} ({len(train_df):,} rows)")
    print(f"      Distribution: {train_df['language'].value_counts().to_dict()}")
    print(f"      Labels: {train_df['label'].value_counts().to_dict()}")
    print(f"      Code-mixed: {train_df['is_code_mixed'].sum():,} ({train_df['is_code_mixed'].mean()*100:.1f}%)")
    print(f"    - {test_path} ({len(test_df):,} rows)")
    print(f"      Distribution: {test_df['language'].value_counts().to_dict()}")
    print(f"      Labels: {test_df['label'].value_counts().to_dict()}")
    print(f"      Code-mixed: {test_df['is_code_mixed'].sum():,} ({test_df['is_code_mixed'].mean()*100:.1f}%)")

if __name__ == "__main__":
    main()
