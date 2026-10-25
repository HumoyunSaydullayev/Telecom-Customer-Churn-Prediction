"""
views.py — Telekom Churn Tahlili

Ishlash tartibi:
1) train_models.py yoki auto_train — modellarni BIR MARTA o'qitadi (static/models/ ga saqlaydi)
2) Foydalanuvchi YANGI CSV yuklaydi → saqlangan model PREDICT qiladi (qayta train QILMAYDI)
3) Agar Churn ustuni bo'lsa → aniqlik metrikalarini ko'rsatadi
"""

import os
import io
import base64
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Model yuklash yordamchilari
# ─────────────────────────────────────────────────────────────────────────────

def get_models_dir():
    path = os.path.join(settings.BASE_DIR, 'static', 'models')
    os.makedirs(path, exist_ok=True)
    return path


def load_classifier(model_choice):
    model_map = {'rf': 'rf_model.pkl', 'gbm': 'gbm_model.pkl', 'lr': 'lr_model.pkl'}
    fname = model_map.get(model_choice)
    if fname is None:
        return None
    path = os.path.join(get_models_dir(), fname)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_kmeans(k):
    path = os.path.join(get_models_dir(), f'kmeans_k{k}.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def models_exist():
    d = get_models_dir()
    return all(os.path.exists(os.path.join(d, f))
               for f in ['rf_model.pkl', 'gbm_model.pkl', 'lr_model.pkl'])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_telcom(df):
    """Telekom datasetni ML uchun tayyorlash — NaN muammosi to‘liq hal qilindi"""
    df = df.copy()

    # TotalCharges ni to‘g‘ri tozalash (pandas CoW muammosi tuzatildi)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Churn ustuni
    target_col = next((c for c in df.columns if c.lower() == 'churn'), None)
    if target_col:
        col = df[target_col].astype(str).str.strip().str.lower()
        df[target_col] = col.map(
            {'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0}
        ).fillna(0).astype(int)

    # ID ustunlarini o‘chirish
    drop_cols = [c for c in df.columns if 'id' in c.lower()]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Kategorik ustunlarni kodlash
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Qolgan barcha NaN larni 0 bilan to‘ldirish (xavfsizlik)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df, target_col


# ─────────────────────────────────────────────────────────────────────────────
# Auto-train (BIR MARTA — faqat pkl lar yo'q bo'lganda)
# ─────────────────────────────────────────────────────────────────────────────

def auto_train_all_models(csv_path=None):
    if csv_path is None:
        possible_paths = [
            os.path.join(settings.BASE_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
            os.path.join(settings.BASE_DIR, 'static', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        ]
        csv_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if csv_path is None:
            print("[auto_train] Default CSV topilmadi!")
            return False

    try:
        df = pd.read_csv(csv_path)
        df_proc, target_col = preprocess_telcom(df)          # ← Yangi preprocess ishlatilmoqda

        X = df_proc.drop(columns=[target_col])
        y = df_proc[target_col]
        feature_names = list(X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        models_dir = get_models_dir()

        # Random Forest
        rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        joblib.dump({'model': rf, 'scaler': scaler, 'feature_names': feature_names,
                     'accuracy': rf.score(X_test, y_test)},
                    os.path.join(models_dir, 'rf_model.pkl'))

        # Gradient Boosting (Endi ishlaydi!)
        gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        gbm.fit(X_train, y_train)
        joblib.dump({'model': gbm, 'scaler': scaler, 'feature_names': feature_names,
                     'accuracy': gbm.score(X_test, y_test)},
                    os.path.join(models_dir, 'gbm_model.pkl'))

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        joblib.dump({'model': lr, 'scaler': scaler, 'feature_names': feature_names,
                     'accuracy': lr.score(X_test, y_test)},
                    os.path.join(models_dir, 'lr_model.pkl'))

        # KMeans
        CLUSTER_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        available = [f for f in CLUSTER_FEATURES if f in df_proc.columns]
        X_cluster = df_proc[available].fillna(0)
        scaler_km = StandardScaler()
        X_km_scaled = scaler_km.fit_transform(X_cluster)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X_km_scaled)
        explained = f"{sum(pca.explained_variance_ratio_) * 100:.1f}%"

        for k in [2, 3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_km_scaled)
            joblib.dump({
                'kmeans': km, 'scaler': scaler_km, 'pca': pca,
                'cluster_features': available, 'explained_variance': explained,
            }, os.path.join(models_dir, f'kmeans_k{k}.pkl'))

        print("[auto_train] ✅ BARCHA MODELLAR MUVAFFAKIYATLI SAQLANDI!")
        return True

    except Exception as e:
        print(f"[auto_train] Xatolik: {e}")
        return False

def ensure_models_exist():
    if not models_exist():
        print("🔄 Modellar topilmadi → avtomatik o‘qitilmoqda...")
        auto_train_all_models()


# ─────────────────────────────────────────────────────────────────────────────
# Grafik yordamchilari
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_confusion_matrix(cm, labels=('No Churn', 'Churn')):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Bashorat qilingan', fontsize=11)
    ax.set_ylabel('Haqiqiy', fontsize=11)
    ax.set_title('Chalkashlik Matritsasi', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_feature_importance(feature_names, importances, top_n=15):
    top_n = min(top_n, len(feature_names))
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, top_n))
    ax.barh(range(top_n), importances[idx][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Muhimlik darajasi", fontsize=11)
    ax.set_title("Eng muhim xususiyatlar (Feature Importance)", fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#0f4c75', lw=2,
            label=f"ROC egri chizig'i (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Yolg'on ijobiy darajasi (FPR)", fontsize=11)
    ax.set_ylabel("To'g'ri ijobiy darajasi (TPR)", fontsize=11)
    ax.set_title("ROC Egri Chizig'i", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_churn_distribution(y):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts()
    ax.pie(counts, labels=['Qoladi (No Churn)', 'Ketadi (Churn)'],
           colors=['#2a9d8f', '#e76f51'], autopct='%1.1f%%',
           startangle=90, wedgeprops=dict(width=0.6))
    ax.set_title("Mijozlar taqsimoti", fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_tenure_churn(df, target_col):
    if 'tenure' not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, label, color in [(0, "Qoladi", "#2a9d8f"), (1, "Ketadi", "#e76f51")]:
        ax.hist(df[df[target_col] == val]['tenure'], bins=30,
                alpha=0.6, label=label, color=color)
    ax.set_xlabel("Abonentlik muddati (oy)", fontsize=11)
    ax.set_ylabel("Mijozlar soni", fontsize=11)
    ax.set_title("Abonentlik muddati va Churn", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_prediction_distribution(y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    unique, counts = np.unique(y_pred, return_counts=True)
    labels = ['Qoladi' if v == 0 else 'Ketadi' for v in unique]
    ax.pie(counts, labels=labels,
           colors=['#2a9d8f', '#e76f51'], autopct='%1.1f%%',
           startangle=90, wedgeprops=dict(width=0.6))
    ax.set_title("Bashorat natijalari taqsimoti", fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_probability_histogram(y_prob):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_prob, bins=40, color='#0f4c75', alpha=0.7, edgecolor='white')
    ax.axvline(x=0.5, color='#ef4444', linestyle='--', lw=2, label='Chegara (0.5)')
    ax.set_xlabel("Churn ehtimolligi", fontsize=11)
    ax.set_ylabel("Mijozlar soni", fontsize=11)
    ax.set_title("Churn Ehtimollik Taqsimoti", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_cluster_scatter(X_pca, labels, n_clusters):
    colors = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#f4a261']
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=15, alpha=0.6, color=colors[i % len(colors)],
                   label=f'Klaster {i}')
    ax.set_xlabel('PCA Komponent 1', fontsize=11)
    ax.set_ylabel('PCA Komponent 2', fontsize=11)
    ax.set_title('Mijoz klasterlari (PCA proeksiyasi)', fontsize=13, fontweight='bold')
    ax.legend(markerscale=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_cluster_bar(cluster_info):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"Klaster {c['id']}" for c in cluster_info]
    counts = [c['count'] for c in cluster_info]
    colors = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#f4a261']
    bars = ax.bar(names, counts, color=colors[:len(names)],
                  edgecolor='white', linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, str(count),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mijozlar soni', fontsize=11)
    ax.set_title("Klasterlardagi mijozlar soni", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_cluster_heatmap(cluster_means, feature_subset):
    fig, ax = plt.subplots(figsize=(10, 4))
    data = cluster_means[feature_subset].T
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Klaster xususiyatlari o'rtacha qiymatlari",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Klaster ID", fontsize=11)
    plt.tight_layout()
    return fig_to_base64(fig)


def assign_cluster_label(row):
    tenure = row.get('tenure', 10)
    monthly = row.get('MonthlyCharges', 50)
    churn = row.get('Churn', 0.5)
    if churn > 0.5:
        return "Yuqori xavfli premium mijozlar" if monthly > 70 else "Churn xavfi yuqori mijozlar"
    elif tenure > 30 and monthly < 50:
        return "Sodiq tejamkor mijozlar"
    elif monthly > 70:
        return "Premium mijozlar"
    return "O'rtacha faol mijozlar"


# ─────────────────────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────────────────────

def index(request):
    ensure_models_exist()
    return render(request, 'index.html')


def database(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        saved = fs.save(f"{uuid.uuid4()}_{csv_file.name}", csv_file)
        file_path = fs.path(saved)

        try:
            df = pd.read_csv(file_path)
            sample_html = df.head(20).to_html(
                index=False,
                classes='table table-striped table-hover table-bordered',
                border=0
            )
            churn_col = next((c for c in df.columns if c.lower() == 'churn'), None)
            if churn_col:
                churn_vals = df[churn_col].astype(str).str.strip().str.lower()
                churn_count = churn_vals.isin(['yes', '1', 'true', '1.0']).sum()
                churn_rate = f"{churn_count / len(df) * 100:.1f}%"
            else:
                churn_rate = "N/A"

            context.update({
                'sample_data': sample_html,
                'total_rows': len(df),
                'total_cols': len(df.columns),
                'churn_rate': churn_rate,
                'uploaded': True,
                'file_name': csv_file.name,
            })
        except Exception as e:
            context['error'] = f"Xatolik: {str(e)}"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        sample = {
            'customerID':      ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK', '7795-CFOCW', '9237-HQITU'],
            'gender':          ['Female', 'Male', 'Male', 'Male', 'Female'],
            'SeniorCitizen':   [0, 0, 0, 0, 0],
            'tenure':          [1, 34, 2, 45, 2],
            'Contract':        ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month'],
            'MonthlyCharges':  [29.85, 56.95, 53.85, 42.30, 70.70],
            'TotalCharges':    [29.85, 1889.50, 108.15, 1840.75, 151.65],
            'InternetService': ['DSL', 'DSL', 'DSL', 'DSL', 'Fiber optic'],
            'PaymentMethod':   ['Electronic check', 'Mailed check', 'Mailed check',
                                'Bank transfer', 'Electronic check'],
            'Churn':           ['No', 'No', 'Yes', 'No', 'Yes'],
        }
        sample_html = pd.DataFrame(sample).to_html(
            index=False,
            classes='table table-striped table-hover table-bordered',
            border=0
        )
        context.update({
            'sample_data': sample_html,
            'total_rows': 7043,
            'total_cols': 21,
            'churn_rate': '26.5%',
        })

    return render(request, 'database.html', context)


def algorithm(request):
    return render(request, 'algorithm.html')


def analysis(request):
    """
    Klassifikatsiya sahifasi:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    MUHIM: Saqlangan model FAQAT PREDICT qiladi!
    CSV yuklanganda qayta train QILMAYDI.

    Ishlash tartibi:
    1) Saqlangan model yuklanadi (static/models/*.pkl)
    2) Yuklangan CSV preprocesssing qilinadi
    3) Feature nomlarini saqlangan model bilan moslashtiradi
    4) Model.predict() — faqat bashorat
    5) Agar CSV da Churn ustuni bo'lsa → haqiqiy vs bashorat solishtiriladi
    6) Agar Churn ustuni yo'q bo'lsa → faqat bashorat natijalari ko'rsatiladi
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    context = {}

    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        model_choice = request.POST.get('model_choice', 'rf')

        if not csv_file:
            context['error'] = "Iltimos, CSV fayl yuklang."
            return render(request, 'analysis.html', context)

        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        saved = fs.save(f"{uuid.uuid4()}_{csv_file.name}", csv_file)
        file_path = fs.path(saved)

        try:
            df_raw = pd.read_csv(file_path)
            df_proc, target_col = preprocess_telcom(df_raw)

            has_churn = target_col is not None and target_col in df_proc.columns
            if has_churn:
                y_true = df_proc[target_col]
                X_all = df_proc.drop(columns=[target_col])
            else:
                y_true = None
                X_all = df_proc

            feature_names = list(X_all.columns)

            # ─── Saqlangan modelni yuklash ─────────────────────────────────
            saved_data = load_classifier(model_choice)

            if saved_data is None:
                # Model topilmadi — auto_train qilish
                ensure_models_exist()
                saved_data = load_classifier(model_choice)

            if saved_data is None:
                context['error'] = (
                    "Model topilmadi! Avval train_models.py ni ishga tushiring yoki "
                    "asosiy datasetni (WA_Fn-UseC_-Telco-Customer-Churn.csv) "
                    "manage.py yoniga joylashtiring."
                )
                return render(request, 'analysis.html', context)

            model = saved_data['model']
            scaler = saved_data['scaler']
            saved_features = saved_data.get('feature_names', [])
            saved_accuracy = saved_data.get('accuracy', None)

            # ─── Feature moslashtirish ─────────────────────────────────────
            # Yuklangan CSV dagi feature lar saqlangan model feature lariga moslashtiriladi
            missing_in_csv = set(saved_features) - set(feature_names)
            extra_in_csv = set(feature_names) - set(saved_features)

            if missing_in_csv:
                # Yetishmayotgan ustunlarni 0 bilan to'ldirish
                for col_name in missing_in_csv:
                    X_all[col_name] = 0

            # Faqat saqlangan model kutgan ustunlarni, to'g'ri tartibda olish
            X_ordered = X_all[saved_features]
            X_scaled = scaler.transform(X_ordered.values)

            # ─── FAQAT PREDICT (train QILMAYDI!) ──────────────────────────
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]

            model_names = {
                'rf':  'Random Forest',
                'gbm': 'Gradient Boosting',
                'lr':  'Logistik Regressiya',
            }
            model_name = model_names.get(model_choice, model_choice)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.abs(model.coef_[0])

            # Grafiklar — doim ko'rsatiladi
            fi_img = plot_feature_importance(np.array(saved_features), importances)
            pred_dist_img = plot_prediction_distribution(y_pred)
            prob_hist_img = plot_probability_histogram(y_prob)

            churn_predicted = int(y_pred.sum())
            no_churn_predicted = int(len(y_pred) - churn_predicted)

            context.update({
                'results':            True,
                'model_name':         f"{model_name} (saqlangan model)",
                'total_records':      len(df_raw),
                'n_features':         len(saved_features),
                'churn_predicted':    churn_predicted,
                'no_churn_predicted': no_churn_predicted,
                'fi_img':             fi_img,
                'pred_dist_img':      pred_dist_img,
                'prob_hist_img':      prob_hist_img,
            })

            if saved_accuracy:
                context['train_accuracy'] = f"{saved_accuracy * 100:.2f}"

            # ─── Agar Churn ustuni bo'lsa → metrikalar hisobi ─────────────
            if has_churn and y_true is not None:
                accuracy = accuracy_score(y_true, y_pred)
                roc_auc_val = roc_auc_score(y_true, y_prob)
                cm = confusion_matrix(y_true, y_pred)
                report_dict = classification_report(y_true, y_pred, output_dict=True)

                cm_img = plot_confusion_matrix(cm)
                roc_img = plot_roc(y_true, y_prob)
                dist_img = plot_churn_distribution(y_true)
                tenure_img = plot_tenure_churn(df_proc, target_col)

                report_rows = []
                for label, metrics in report_dict.items():
                    if label in ('accuracy', 'macro avg', 'weighted avg'):
                        continue
                    report_rows.append({
                        'label':     'Churn' if str(label) == '1' else 'No Churn',
                        'precision': f"{metrics['precision']:.3f}",
                        'recall':    f"{metrics['recall']:.3f}",
                        'f1_score':  f"{metrics['f1-score']:.3f}",
                        'support':   int(metrics['support']),
                    })

                context.update({
                    'has_churn_col':  True,
                    'accuracy':       f"{accuracy * 100:.2f}",
                    'roc_auc':        f"{roc_auc_val:.4f}",
                    'churn_count':    int(y_true.sum()),
                    'cm_img':         cm_img,
                    'roc_img':        roc_img,
                    'dist_img':       dist_img,
                    'tenure_img':     tenure_img,
                    'report_rows':    report_rows,
                })
            else:
                context['has_churn_col'] = False

            if missing_in_csv:
                context['warning'] = (
                    f"CSV da ba'zi ustunlar topilmadi va 0 bilan to'ldirildi: "
                    f"{', '.join(sorted(missing_in_csv)[:5])}"
                    + (f" va yana {len(missing_in_csv)-5} ta" if len(missing_in_csv) > 5 else "")
                )
            if extra_in_csv:
                context['info'] = (
                    f"CSV dagi qo'shimcha ustunlar e'tiborga olinmadi: "
                    f"{', '.join(sorted(extra_in_csv)[:5])}"
                    + (f" va yana {len(extra_in_csv)-5} ta" if len(extra_in_csv) > 5 else "")
                )

        except Exception as e:
            context['error'] = f"Xatolik: {str(e)}"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return render(request, 'analysis.html', context)


def klassterlash_view(request):
    """
    Klasterlash sahifasi:
    Saqlangan KMeans model bilan yangi CSV ni klasterlaydi (qayta train QILMAYDI).
    """
    context = {}
    CLUSTER_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        n_clusters = int(request.POST.get('n_clusters', 3))
        n_clusters = max(2, min(5, n_clusters))

        if not csv_file:
            context['error'] = "Iltimos, CSV fayl yuklang."
            return render(request, 'klassterlash.html', context)

        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        saved = fs.save(f"{uuid.uuid4()}_{csv_file.name}", csv_file)
        file_path = fs.path(saved)

        try:
            df = pd.read_csv(file_path)
            df_proc, target_col = preprocess_telcom(df)

            available = [f for f in CLUSTER_FEATURES if f in df_proc.columns]
            if len(available) < 2:
                available = df_proc.columns[:4].tolist()
                if target_col and target_col in available:
                    available.remove(target_col)

            X_cluster = df_proc[available].fillna(0)

            # ─── Saqlangan KMeans modelni yuklash ──────────────────────────
            saved_km = load_kmeans(n_clusters)

            if saved_km is None:
                ensure_models_exist()
                saved_km = load_kmeans(n_clusters)

            if saved_km:
                scaler_km = saved_km['scaler']
                pca = saved_km['pca']
                kmeans = saved_km['kmeans']
                saved_cluster_features = saved_km.get('cluster_features', available)

                # Feature moslashtirish
                for f in saved_cluster_features:
                    if f not in df_proc.columns:
                        df_proc[f] = 0
                X_cluster = df_proc[saved_cluster_features].fillna(0)

                X_scaled = scaler_km.transform(X_cluster)
                X_pca = pca.transform(X_scaled)
                labels = kmeans.predict(X_scaled)
                explained = saved_km['explained_variance']
                km_source = "saqlangan model"
                available = saved_cluster_features
            else:
                # Model topilmadi — yangi train
                scaler_km = StandardScaler()
                X_scaled = scaler_km.fit_transform(X_cluster)
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                explained = f"{sum(pca.explained_variance_ratio_) * 100:.1f}"
                km_source = "yangi train (model topilmadi)"

                joblib.dump({
                    'kmeans': kmeans, 'scaler': scaler_km,
                    'pca': pca, 'cluster_features': available,
                    'explained_variance': explained,
                }, os.path.join(get_models_dir(), f'kmeans_k{n_clusters}.pkl'))

            df_proc['cluster'] = labels

            # Churn ustunini qo'shish (agar mavjud bo'lsa)
            mean_cols = available[:]
            if target_col and target_col in df_proc.columns:
                mean_cols.append(target_col)

            cluster_means = df_proc.groupby('cluster')[mean_cols].mean()
            cluster_counts = df_proc['cluster'].value_counts().sort_index()

            cluster_info = []
            for i in range(n_clusters):
                if i in cluster_means.index:
                    row = cluster_means.loc[i].to_dict()
                else:
                    row = {}
                cluster_info.append({
                    'id':          i,
                    'label':       assign_cluster_label(row),
                    'count':       int(cluster_counts.get(i, 0)),
                    'percentage':  f"{cluster_counts.get(i, 0) / len(df_proc) * 100:.1f}",
                    'churn_rate':  f"{row.get(target_col, 0) * 100:.1f}" if target_col else "N/A",
                    'avg_tenure':  f"{row.get('tenure', 0):.1f}",
                    'avg_monthly': f"{row.get('MonthlyCharges', 0):.2f}",
                })

            scatter_img = plot_cluster_scatter(X_pca, labels, n_clusters)
            bar_img = plot_cluster_bar(cluster_info)
            heat_cols = [c for c in available if c != target_col]
            heatmap_img = plot_cluster_heatmap(cluster_means, heat_cols)

            context.update({
                'results':            True,
                'n_clusters':         n_clusters,
                'total_records':      len(df),
                'cluster_info':       cluster_info,
                'scatter_img':        scatter_img,
                'bar_img':            bar_img,
                'heatmap_img':        heatmap_img,
                'features_used':      available,
                'explained_variance': explained,
                'km_source':          km_source,
            })

        except Exception as e:
            context['error'] = f"Xatolik: {str(e)}"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return render(request, 'klassterlash.html', context)