# مرجع الباراميترات: من v3 إلى v7

هذا الملف يوثّق كل الباراميترات المهمة اللي جرّبناها، القيم اللي اشتغلت والقيم اللي كسرت الموديل، والتعارضات اللي لازم تعرفها.

## جدول التطوّر الكامل

| الباراميتر | v3 | v4 | v5 | v6 | v7 | الأفضل |
|-----------|:--:|:--:|:--:|:--:|:--:|:------:|
| **Bottleneck** | 128 | 128 | 128 | 128 | 128 | 128 |
| **Activation** | SELU ❌ | ELU | ELU | ELU | ELU | **ELU** |
| **Kernel init** | lecun_normal | he_normal | he_normal | he_normal | he_normal | **he_normal** |
| **BatchNorm** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Dropout type** | AlphaDropout | MCDropout ❌ | Dropout | Dropout | Dropout | **Dropout عادي** |
| **Dropout on bottleneck** | 0.3 | 0.3 | 0.15 | 0.15 | 0.15 | **0.15** |
| **Dropout on head** | 0.2 | 0.2 | 0.2-0.3 | 0.20-0.30 | 0.25-0.35 | **0.30** |
| **Loss (Phase 1)** | Focal(α=0.25) | Focal(α=0.25) | CrossEntropy | CrossEntropy | CrossEntropy | **CrossEntropy** |
| **Loss (Phase 2)** | Focal(α=0.25) | Focal(α=0.25) | Focal(α=1.0) | CrossEntropy | CrossEntropy | **CrossEntropy** |
| **Head units** | [64] | [64] | [256, 128] | [512, 256] | متعدد | **[512, 256]** |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | **AdamW** |
| **LR (Phase 1)** | 1e-3 | 3e-3 | 1e-3 | 1e-3 | 1e-3 | **1e-3** |
| **LR (Phase 2)** | 5e-5 | 5e-4 | 3e-4 | 3e-4 | 1e-5 | **3e-4** |
| **Weight decay** | ❌ | ❌ | ❌ | 1e-4 | 1e-4 | **1e-4** |
| **LR schedule** | ExpDecay | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | **ReduceLROnPlateau** |
| **LR factor** | 0.5 | 0.3 | 0.3 | 0.3 | 0.3 | **0.3** |
| **LR patience** | - | 5 | 5 | 6 | 6 | **6** |
| **ES patience (val_acc)** | 12 | 15 | 15 | 40 | 40 | **40** |
| **ES patience (loss)** | 10 | 10 | 10 | 25 | 25 | **25** |
| **Augmentation** | كثيف | كثيف | خفيف | خفيف مضبوط | خفيف | **خفيف مضبوط** |
| **horizontal_flip** | ✅ | ✅ | ❌ | ❌ | ❌ | **❌** |
| **rotation_range** | 10 | 10 | 8 | 8 | 8 | **8** |
| **Encoder trainable** | مجمّد | مجمّد | مجمّد | مجمّد | آخر block | **مجمّد** |
| **Val size** | 120 | 120 | 120 | 240 | 240 | **240** |
| **class_weight** | ❌ | ❌ | ❌ | Shirt=2.5 | Shirt=2.5 | **مفعّل** |
| **sample_weight (Pseudo)** | ❌ | ❌ | 3.0/1.0 | 3.0/1.0 | 3.0/1.0 | **3.0/1.0** |
| **Uncertainty threshold** | P40 | P40 | P20 | P20 | P20 | **P20** |
| **Confidence threshold** | 0.85 | 0.85 | 0.95 | 0.95 | 0.95 | **0.95** |
| **Pseudo-labeling gate** | ❌ | ❌ | 72% | 78% | 78% | **78%** |
| **Head Sweep** | ❌ | ❌ | ❌ | ✅ | ❌ | **✅** |
| **Ensemble** | ❌ | ❌ | ❌ | ❌ | ✅ (فشل) | tbd |
| **TTA** | ❌ | ❌ | ❌ | ❌ | ✅ (فشل) | tbd |
| **النتيجة** | 11% | 42% | 70% | **83.66%** ⭐ | 75.6% | v6 |

---

## أفضل التركيبات (اللي تشتغل مع بعض)

### التركيبة v6 — 83.66%

```python
# ─── Encoder (مجمّد) ───
Activation:     ELU
Kernel init:    he_normal
BatchNorm:      ✅ (بعد كل Conv, قبل ELU)
Bottleneck:     Dense(128, kernel_constraint=max_norm(3.0))

# ─── Head ───
Structure:      Dropout(0.15) → Dense(512)+BN → Dropout(0.30)
                             → Dense(256)+BN → Dropout(0.30)
                             → Dense(10, softmax)

# ─── التدريب ───
Optimizer:      AdamW(lr=1e-3, weight_decay=1e-4)
Loss:           CrossEntropy
Schedule:       ReduceLROnPlateau(factor=0.3, patience=6)
Early stop:     val_accuracy patience=40, loss patience=25

# ─── Augmentation ───
rotation_range:      8
zoom_range:          0.1
width/height_shift:  0.1
horizontal_flip:     ❌ (مهم — انظر التعارضات)

# ─── Class weights (لإنقاذ Shirt) ───
Shirt (6):           2.5
T-shirt/Pull/Coat:   1.5
others:              1.0

# ─── Pseudo-labeling ───
Gate:                test_acc >= 0.78
Uncertainty:         < P20 (20th percentile)
Confidence:          > 0.95
sample_weight:       labeled=3.0, pseudo=1.0
Rollback guard:      إذا test_acc نزل → ارجع لـ Phase 1
```

### لماذا هذه التركيبة تشتغل؟

1. **ELU + he_normal + BatchNorm** — الثلاثة يشتغلون مع بعض تاريخياً وتم اختبارهم آلاف المرات
2. **Dropout عادي (مش MCDropout)** — يطفى تلقائياً وقت التقييم
3. **CrossEntropy** — إشارة gradient قوية وواضحة، Focal يعقّد بدون فائدة هنا
4. **AdamW + weight_decay** — regularization أفضل من L2 التقليدي
5. **ReduceLROnPlateau** — يخفض الـ LR فقط لما التحسّن يقف فعلاً (أذكى من ExponentialDecay)
6. **patience كبيرة** — يعطي الموديل وقت كافي للتقارب مع الـ augmentation
7. **class_weight** — يجبر الموديل يهتم بـ Shirt بدل ما يتجاهله

---

## التعارضات: باراميترات لا يجب أن تتواجد مع بعض

### ❌ التعارض 1: SELU + BatchNorm

**ما جرّبناه في v3:** SELU مع BatchNorm.

**السبب في الفشل:** SELU مصمّم ليكون *self-normalizing* — يحافظ تلقائياً على `mean≈0` و `variance≈1` عبر الشبكة. BatchNorm يفرض توزيعاً مختلفاً على كل batch، وهذا يكسر خاصية SELU الأساسية.

**النتيجة:** الشبكة ما تستفيد لا من SELU (لأن BN كسرت normalization) ولا من BN (لأن SELU لا يحتاجها). التدريب انهار لـ 11%.

**القاعدة:**
```
if activation == 'selu':
    ❌ لا تستخدم BatchNorm
    ✅ استخدم AlphaDropout (مش Dropout عادي)
    ✅ Kernel init: lecun_normal

if activation in ('relu', 'elu'):
    ✅ استخدم BatchNorm
    ✅ استخدم Dropout عادي
    ✅ Kernel init: he_normal
```

### ❌ التعارض 2: MCDropout في وضع التقييم

**ما جرّبناه في v4:** استخدام `MCDropout` (dropout دائم التفعيل) في كل الشبكة.

**السبب في الفشل:** MCDropout يجبر `training=True` دائماً، حتى وقت الـ validation والـ test. النتيجة: كل قياس دقة يتم مع 30% من الـ features مطفية عشوائياً. الـ EarlyStopping اعتمد على أرقام ضوضائية.

**القاعدة:**
```
✅ استخدم Dropout عادي في الشبكة كلها
✅ لعدم اليقين: فعّل training=True يدوياً فقط عند الحاجة
   mc = np.stack([model(X, training=True) for _ in range(50)])
❌ لا تستبدل Dropout كله بـ MCDropout
```

### ❌ التعارض 3: Focal Loss في المرحلة الأولى

**ما جرّبناه في v3/v4:** Focal Loss مع `alpha=0.25` من البداية.

**السبب في الفشل:** Focal Loss تصميمها للـ hard examples، لكن في البداية كل الأمثلة صعبة (الموديل عشوائي). `alpha=0.25` يقلل الـ gradient 4×، و `(1-p)^γ` مع `p≈0.1` يعطي weight ثابت تقريباً. النتيجة: gradient ضعيف جداً، ما في تعلّم.

**القاعدة:**
```
المرحلة 1 (تدريب من الصفر):
  ✅ CrossEntropy — إشارة قوية

المرحلة 2 (بيانات فيها ضوضاء / pseudo labels):
  ✅ Focal Loss(gamma=1.5, alpha=1.0) — يركز على الأخطاء

❌ لا تستخدم alpha < 1.0 إلا إذا في تخصيص واضح للأصناف
❌ لا تستخدم Focal في البداية إذا الموديل ضعيف
```

### ❌ التعارض 4: horizontal_flip + Fashion MNIST

**ما جرّبناه في v3/v4:** `horizontal_flip=True` كجزء من Augmentation.

**السبب في الفشل:** Fashion MNIST فيه أحذية موجّهة (Sandal, Sneaker, Ankle boot) — كلها تنظر لجهة اليسار. القلب الأفقي يُنشئ صور "أحذية تنظر لليمين" لا توجد في التوزيع الأصلي. الموديل يتعلم على صور خارج التوزيع ويُختبر على صور داخله.

**القاعدة:**
```
✅ حلّل التوزيع قبل الـ augmentation
✅ Fashion MNIST: rotation, zoom, shift فقط
❌ لا تستخدم horizontal_flip مع بيانات موجّهة
   (أحذية، نصوص، أسهم، وجوه بجانب واحد)
```

### ❌ التعارض 5: Data Augmentation + Encoder مجمّد

**ما لاحظناه في v5:** Augmentation قوي مع encoder مجمّد أعطى نتائج أسوأ من بدون augmentation.

**السبب:** BatchNorm داخل الـ encoder المجمّد حفظت `moving_mean` و `moving_variance` من التدريب الأصلي على صور نظيفة. الصور المُدوّرة/المُزاحة تُطبَّع بإحصاءات خاطئة، فتُسقَط في أماكن غريبة من فضاء الـ 128.

**القاعدة:**
```
Encoder مجمّد:
  ✅ Augmentation خفيف (rotation<10°, zoom<0.1)
  ❌ لا augmentation قوي — يخرّب موقع الصور في الفضاء

Encoder غير مجمّد (fine-tuning):
  ✅ Augmentation قوي مقبول
  ✅ الـ BN تتحدّث مع الصور الجديدة
```

### ❌ التعارض 6: patience صغيرة + Augmentation

**ما جرّبناه في v5:** EarlyStopping بـ `patience=10` على `loss`.

**السبب في الفشل:** الـ augmentation يجعل `train_loss` ضوضائية (كل batch مختلف). `patience=10` قصيرة جداً — التدريب توقف عند epoch 19 قبل التقارب الفعلي.

**القاعدة:**
```
بدون augmentation:
  ✅ patience على loss: 10-15
  ✅ patience على val_acc: 15-20

مع augmentation:
  ✅ patience على loss: 20-30
  ✅ patience على val_acc: 30-40
  ✅ اجعل val_accuracy هو الـ primary monitor
```

### ❌ التعارض 7: Pseudo-labeling من موديل ضعيف

**ما جرّبناه في v4:** pseudo-labeling على موديل دقته 55%.

**السبب في الفشل:** موديل 55% ينتج pseudo labels ~35% منها خاطئة حتى بعد الفلترة. تدريب على 20K صورة فيها 7K غلط مقابل 960 صحيحة = الضوضاء تغرق الإشارة. اسمه *confirmation bias*.

**القاعدة:**
```
Pseudo-labeling gate:
  ✅ ابدأ فقط إذا test_acc >= 78%
  ✅ استخدم threshold صارم (P20 + confidence > 0.95)
  ✅ ضع rollback guard — لو تراجعت الدقة ارجع للنسخة السابقة
  ✅ استخدم sample_weight لإعطاء الـ labels الحقيقية وزن أعلى (3×)
  ❌ لا تعمل pseudo-labeling إذا الموديل ضعيف
```

### ❌ التعارض 8: مشاركة encoder بين موديلات ensemble

**ما جرّبناه في v7:** استخدام نفس `encoder_base` لثلاثة موديلات مختلفة مع محاولة "استرجاع" الأوزان بين كل موديل.

**السبب في الفشل:** الـ `encoder_base` كان **نفس الكائن في الذاكرة** لكل الموديلات. عندما دُرِّب M1 عدّل الأوزان، ولما بدأ M2 حُمِّلت الأوزان الأصلية للـ Conv layers لكن الـ BatchNorm `moving_statistics` بقيت متأثرة. الموديلات صارت غير متزامنة داخلياً.

**القاعدة:**
```
Ensemble مع encoder مشترك:
  ✅ انسخ الـ encoder بالكامل لكل موديل: keras.models.clone_model(encoder)
  ✅ أو ابنِ كل encoder من الصفر واحمّل الأوزان من ملف
  ❌ لا تشارك نفس Model instance بين موديلات متعددة تُدرَّب

الحل الأفضل: احفظ encoder على disk، حمّله لكل موديل جديد:
   encoder_i = keras.models.load_model("encoder.keras")
```

---

## قواعد عامة للاختيار

### الـ Activation
- **ReLU** — الافتراضي، سريع، يشتغل مع أي شي
- **ELU** — أفضل لـ Conv deep، يحتفظ بمعلومات القيم السالبة
- **SELU** — فقط إذا الشبكة كلها Dense (بدون BN، بدون Dropout عادي)

### الـ Optimizer
- **Adam** — الافتراضي
- **AdamW** — أفضل مع weight_decay صريح (1e-4 عادةً)
- **SGD + momentum** — أدق للـ fine-tuning لكن أبطأ

### الـ Learning Rate
- **1e-3** — بداية قياسية للشبكات الجديدة
- **5e-4 إلى 3e-4** — للـ heads فوق encoder مجمّد
- **1e-5** — للـ fine-tuning (يحمي ما تعلّمه الـ encoder)
- **> 5e-3** — نادراً، يحتاج warmup

### الـ LR Schedule
- **ReduceLROnPlateau** — الأذكى، يتفاعل مع الأداء الفعلي
- **CosineDecay** — كويس للـ fine-tuning
- **ExponentialDecay** — بسيط لكن أعمى
- **Step decay** — قديم، تجنّبه

### الـ Regularization
- **Dropout 0.2-0.3** — قياسي للـ Dense layers
- **BatchNorm** — بعد كل Conv/Dense، قبل الـ activation
- **weight_decay 1e-4** — مع AdamW
- **max_norm(3.0)** — على الـ bottleneck خاصة

### الـ Batch Size
- **32** — للـ head الصغير (بيانات قليلة)
- **64-128** — للـ pseudo-labeling (بيانات أكثر)
- **256** — للـ Autoencoder (بيانات كثيرة، صور صغيرة)
- **> 512** — يحتاج LR أعلى، تجنّب مع 960 صورة فقط

---

## قائمة فحص قبل التدريب

قبل ما تشغّل أي تجربة جديدة، اسأل:

- [ ] الـ activation والـ initializer متوافقين؟ (ELU+he, SELU+lecun)
- [ ] الـ Dropout الافتراضي (مش MCDropout) في وضع التقييم؟
- [ ] الـ augmentation ما فيه horizontal_flip إذا البيانات موجّهة؟
- [ ] الـ patience كافية للـ augmentation (25+ على loss)؟
- [ ] الـ pseudo-labeling عنده gate + rollback guard؟
- [ ] الـ encoder المجمّد ما ينشارك بين موديلات ensemble؟
- [ ] الـ val set كبير كفاية للاختيار الموثوق (≥ 200 صورة)؟
- [ ] الـ class_weight مضبوطة للأصناف الصعبة؟
- [ ] الـ saved_models القديمة محذوفة قبل تجربة بنية جديدة؟

---

## ملخص الدروس

1. **الأدوات المتوافقة > الأدوات الفانسي**: ELU + BN + Dropout عادي > SELU + BN
2. **Linear Probe يوفّر ساعات**: يخبرك أين المشكلة قبل ما تجرّب
3. **Head Sweep > التخمين**: 5 تكوينات تُختبَر في دقيقة > ساعة تخمين
4. **حساب Embeddings مسبقاً**: يخلي التجريب أرخص 100×
5. **Pseudo-labeling سلاح ذو حدين**: gate + guard إلزاميّان
6. **augmentation + encoder مجمّد = خفيف فقط**: BN الداخلية حساسة
7. **الاتساق أهم من الطموح**: v6 المستقر > v7 الطموح المكسور
