# 📊 Telecom Customer Churn Prediction & Big Data Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Django-092E20?style=for-the-badge&logo=django&logoColor=white" alt="Django" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%26%20K--Means-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML" />
  <img src="https://img.shields.io/badge/Domain-Big%20Data%20Analytics-blue?style=for-the-badge" alt="Big Data" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Overview / Loyiha haqida

### English
A full-stack Big Data & Machine Learning analytics platform designed to predict customer churn in the telecommunications industry. Using real-world Telco datasets (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), the platform implements classification and customer segmentation (Random Forest, K-Means clustering) wrapped in a responsive Django web interface for customer retention managers.

### O‘zbekcha
Telekommunikatsiya kompaniyalari mijozlarining xizmatlardan voz kechishini (Customer Churn) oldindan bashorat qiluvchi va Big Data tahlilini amalga oshiruvchi to'liq Django veb-platformasi. Tizimda tasniflash (Random Forest) va mijozlarni segmentatsiyalash (K-Means) algoritmlari orqali xavfli mijozlar guruhlari aniqlanadi va tahliliy hisobotlar beriladi.

---

## 🚀 Features / Asosiy imkoniyatlar

- 📈 **Churn Risk Analysis:** Instant inference for individual customer profiles based on tenure, contract type, payment method, and monthly charges.
- 📁 **Batch File Processing:** Upload bulk CSV data (e.g. `test_churn_500.csv`) to assess retention risk across hundreds of subscribers at once.
- 🎯 **Customer Segmentation:** Clustering subscribers into distinct usage profiles.
- 📊 **Visual Dashboard:** Interactive charts, retention metrics, and churn distribution.

---

## 🛠️ Installation & Setup

1. Clone repository:
   ```bash
   git clone https://github.com/HumoyunSaydullayev/Telecom-Customer-Churn-Prediction.git
   cd Telecom-Customer-Churn-Prediction
   ```
2. Install requirements:
   ```bash
   pip install django pandas numpy scikit-learn matplotlib
   ```
3. Run migrations and launch:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
4. Access platform at `http://127.0.0.1:8000/`.

---

## 👨‍💻 Author
**Humoyun Saydullayev**  
Big Data & Machine Learning - KIUT  
GitHub: [@HumoyunSaydullayev](https://github.com/HumoyunSaydullayev)
