# Semi-Supervised Learning — Fashion MNIST

مشروع تصنيف صور Fashion MNIST باستخدام Semi-Supervised Learning: تدريب Autoencoder على 58,800 صورة بدون تسميات، ثم بناء Classifier فوق الـ features المُستخرَجة باستخدام 960 صورة مُسمَّاة فقط.

## النتائج

| النسخة | الوصف | Test Accuracy |
|--------|-------|:-------------:|
| v2 | الأصلية (ELU + Bottleneck 64) | ~78% |
| v3 | SELU + Focal Loss | ❌ 11% |
| v4 | إصلاح v3 | ⚠️ 42% |
| v5 | إصلاح MCDropout | 70% |
| v6 | **Head Sweep + Encoder مجمّد** | **83.66%** ✅ |
| v7 | Fine-tuning + Ensemble + TTA | ⚠️ (يحتاج مراجعة) |

**التوصية:** استخدم **v6** كنسخة إنتاجية — الأدق والأثبت.

## البنية

```
┌─────────────────────────────────────────────────────────────┐
│                    Fashion MNIST (60K + 10K)                │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
    Labeled (1,200)                  Unlabeled (58,800)
        │                                 │
        ├─→ Train (960)                   ├─→ AE Train (56,800)
        └─→ Val (240)                     └─→ AE Val (2,000)

┌─────────────────────────────────────────────────────────────┐
│  المرحلة 0: Autoencoder — Bottleneck 128 (49:1)              │
│  Conv(32)→Conv(64)→Conv(128)→Flatten→Dense(128)              │
│  → Reverse (Decoder)                                         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼ features (128 dim)
┌─────────────────────────────────────────────────────────────┐
│  المرحلة 1: Classifier — Encoder مجمّد                       │
│  Dropout(0.15) → Dense(512) → BN → Dropout(0.30)             │
│                → Dense(256) → BN → Dropout(0.30)             │
│                → Dense(10, softmax)                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  المرحلة 2: Pseudo-Labels                                    │
│  - MC Dropout (50 passes) لقياس عدم اليقين                   │
│  - فلترة: uncertainty < P20 + confidence > 0.95              │
│  - sample_weight: labeled=3.0, pseudo=1.0                    │
│  - حارس تراجع: إذا انخفضت الدقة → استرجع v1                  │
└─────────────────────────────────────────────────────────────┘
```

## الميزات الأساسية

- **Linear Probe تشخيصي**: يقيس جودة الـ AE features قبل التدريب. يجاوب على "هل المشكلة في الـ features ولا في الـ Classifier؟"
- **Head Sweep**: يجرّب 5 تكوينات مختلفة للـ head ويختار الأفضل تجريبياً
- **حساب Embeddings مسبق**: تدريب أسرع 100× (الـ encoder مجمّد فما لزوم لتمرير الصور كل epoch)
- **MC Dropout Calibration**: uncertainty معايرة (فصل +0.14 بين الصحيحة والخاطئة)
- **Pseudo-Labeling بحارس**: بوابة على الدقة الحالية + استرجاع تلقائي عند التراجع

## الإعداد

### المتطلبات

```bash
tensorflow >= 2.13
scikit-learn
numpy
matplotlib
seaborn
```

### المخرجات

```
saved_models/
├── autoencoder.keras        22 MB
├── encoder.keras            3.5 MB
├── decoder.keras            4.1 MB
├── classifier_phase1.keras  5.8 MB
└── classifier_final.keras   5.8 MB

plots/
├── 01_ae_loss.png
├── 02_ae_reconstruction.png
├── 03_head_sweep.png
├── 04_training_history.png
├── 05_confusion_matrix.png
├── 06_uncertainty_correct_vs_wrong.png
└── 07_uncertainty_dist.png
```


## الأداء لكل صنف (v6)

| الصنف | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Trouser | 0.977 | 0.952 | 0.965 |
| Bag | 0.971 | 0.941 | 0.956 |
| Sandal | 0.948 | 0.931 | 0.939 |
| Ankle boot | 0.909 | 0.958 | 0.933 |
| Sneaker | 0.922 | 0.901 | 0.911 |
| Dress | 0.833 | 0.846 | 0.839 |
| T-shirt | 0.792 | 0.821 | 0.806 |
| Coat | 0.694 | 0.736 | 0.714 |
| Pullover | 0.747 | 0.654 | 0.697 |
| Shirt | 0.594 | 0.626 | 0.610 |

**ملاحظة:** الأصناف الأربعة العلوية (T-shirt / Pullover / Coat / Shirt) هي التحدي الأصعب — تتداخل بصرياً في صور 28×28 رمادية.

## هيكل المشروع

```
.
├── README.md                        # هذا الملف
├── PARAMETERS.md                    # شرح الباراميترات والتعارضات
├── .gitignore
├── ssl_fashion_mnist_v6.py          # 
└── plots/                           # الرسوم 
```

## تفاصيل التطوّر

للتفاصيل الكاملة عن كل نسخة والباراميترات، انظر [PARAMETERS.md](PARAMETERS.md).

## المراجع

- الكتاب: *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron (Chapter 10,11,17)
- Fashion MNIST: [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)


