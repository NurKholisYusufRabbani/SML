import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def run_preprocessing(file_path):
    df = pd.read_csv(file_path)
    
    selected_columns = [
        'Hours_Studied', 'Attendance', 'Parental_Involvement', 
        'Access_to_Resources', 'Extracurricular_Activities', 
        'Sleep_Hours', 'Previous_Scores', 'Motivation_Level', 
        'Internet_Access', 'Tutoring_Sessions', 
        'Exam_Score'
    ]
    df = df[selected_columns]
    
    kolom_numerik = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions']
    kolom_kategorikal = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access']
    
    imputer_num = SimpleImputer(strategy='mean')
    df[kolom_numerik] = imputer_num.fit_transform(df[kolom_numerik])
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[kolom_kategorikal] = imputer_cat.fit_transform(df[kolom_kategorikal])
    
    encoder = LabelEncoder()
    for col in kolom_kategorikal:
        df[col] = encoder.fit_transform(df[col])
        
    scaler = StandardScaler()
    df[kolom_numerik] = scaler.fit_transform(df[kolom_numerik])
    
    return df

if __name__ == "__main__":
    print("Memulai proses preprocessing otomatis melalui script...")
    raw_path = 'StudentPerformanceFactors_raw.csv' 
    output_path = 'preprocessing/StudentPerformanceFactors_preprocessing.csv' 
    try:
        df_clean = run_preprocessing(raw_path)
        df_clean.to_csv(output_path, index=False)
        print(f"SUKSES: Data bersih berhasil diproses dan disimpan di {output_path}")
    except Exception as e:
        print(f"GAGAL: Terjadi kesalahan -> {e}")