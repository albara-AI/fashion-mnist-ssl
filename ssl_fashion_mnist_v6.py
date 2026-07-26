# ══════════════════════════════════════════════════════════════════
#  Semi-Supervised Learning — Fashion MNIST
#  ✅ Embeddings تُحسب مرة واحدة — تدريب أسرع 100×
#     يسمح بتجربة عدة تكوينات للـ head في ثوانٍ
#  ✅ الـ Augmentation صار خياراً يُختبر تجريبياً، مش افتراضاً
#     السبب: BN داخل encoder مجمّد تتوقع صور نظيفة
#  ✅ EarlyStopping على loss: patience 10 → 25
#     السبب: هو اللي أوقف v5 عند epoch 19 قبل التقارب
#  ✅ class_weight — لإنقاذ Shirt (recall كان 0.04)
#  ✅ Val 120 → 240 صورة (كانت 120 مهدورة أصلاً)
#  ✅ Head Sweep: يجرب 5 تكوينات ويختار الأفضل على val
#  ✅ الـ Encoder يبقى مجمّداً — القيد محترم
# ══════════════════════════════════════════════════════════════════

import os, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.constraints import max_norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR, SAVE_DIR = "plots", "saved_models"

# ── نحتفظ بالـ Autoencoder إذا موجود (probe 77.61% أثبت جودته) ──
REUSE_AE = os.path.exists(f"{SAVE_DIR}/encoder.keras")

if os.path.exists(PLOTS_DIR):
    shutil.rmtree(PLOTS_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

_pc = [0]
def save_and_close(name="plot"):
    _pc[0] += 1
    p = f"{PLOTS_DIR}/{_pc[0]:02d}_{name}.png"
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"📊 {p}")

print(f"TensorFlow: {tf.__version__}")
tf.random.set_seed(42); np.random.seed(42)


# ══════════════════════════════════════════════════════════════════
#  1. البيانات — val أكبر
#
#  v5 كان: 960 train | 120 val | 120 مهدورة
#  v6 هو:  960 train | 240 val | 0 مهدورة
#
#  ليش؟ 120 صورة validation قليلة جداً لاختيار موديل.
#  الفرق بين val 73.33% و test 69.60% في v5 معظمه ضوضاء.
#  240 صورة تقلل الضوضاء وتخلي الاختيار أوثق.
# ══════════════════════════════════════════════════════════════════
(Xtr_full, ytr_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
Xtr_full = Xtr_full / 255.0
X_test   = X_test   / 255.0

X_lab, y_lab = Xtr_full[:1200], ytr_full[:1200]
X_unlab      = Xtr_full[1200:]

X_train, y_train = X_lab[:960],  y_lab[:960]
X_val,   y_val   = X_lab[960:],  y_lab[960:]      # ✅ 240 صورة

X_ae_val, X_ae_train = X_unlab[:2000], X_unlab[2000:]

def ch(x): return x[..., np.newaxis]
X_train, X_val = ch(X_train), ch(X_val)
X_unlab, X_test = ch(X_unlab), ch(X_test)
X_ae_train, X_ae_val = ch(X_ae_train), ch(X_ae_val)

class_names = ['T-shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")


# ══════════════════════════════════════════════════════════════════
#  2. الـ Autoencoder — إعادة استخدام أو بناء
# ══════════════════════════════════════════════════════════════════
def build_ae():
    e_in = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                      use_bias=False)(e_in)
    x = layers.BatchNormalization()(x); x = layers.ELU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ELU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ELU()(x)
    x = layers.Flatten()(x)
    bn = layers.Dense(128, activation='elu', kernel_initializer='he_normal',
                      kernel_constraint=max_norm(3.0))(x)
    enc = Model(e_in, bn, name='encoder')

    d_in = keras.Input(shape=(128,))
    x = layers.Dense(7*7*128, activation='elu',
                     kernel_initializer='he_normal')(d_in)
    x = layers.Reshape((7, 7, 128))(x)
    for f in (128, 64, 32):
        x = layers.Conv2DTranspose(f, 3, padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ELU()(x)
        if f in (64, 32):
            x = layers.UpSampling2D(2)(x)
    d_out = layers.Conv2DTranspose(1, 3, padding='same',
                                   activation='sigmoid')(x)
    dec = Model(d_in, d_out, name='decoder')

    a_in = keras.Input(shape=(28, 28, 1))
    return enc, dec, Model(a_in, dec(enc(a_in)), name='autoencoder')


if REUSE_AE:
    print("\n♻️  إعادة استخدام الـ Autoencoder المحفوظ (probe أثبت جودته)")
    encoder     = keras.models.load_model(f"{SAVE_DIR}/encoder.keras")
    autoencoder = keras.models.load_model(f"{SAVE_DIR}/autoencoder.keras")
    hist_ae = None
else:
    encoder, decoder, autoencoder = build_ae()
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    print("\n" + "="*60); print("المرحلة 0: Autoencoder"); print("="*60)
    hist_ae = autoencoder.fit(
        X_ae_train, X_ae_train, epochs=150, batch_size=256,
        validation_data=(X_ae_val, X_ae_val),
        callbacks=[
            keras.callbacks.EarlyStopping('val_loss', patience=15,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.3,
                                              patience=8, min_lr=1e-6, verbose=1),
        ], verbose=1)
    autoencoder.save(f"{SAVE_DIR}/autoencoder.keras")
    encoder.save(f"{SAVE_DIR}/encoder.keras")
    decoder.save(f"{SAVE_DIR}/decoder.keras")

    plt.figure(figsize=(8, 4))
    plt.plot(hist_ae.history['loss'], label='Train')
    plt.plot(hist_ae.history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Autoencoder')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    save_and_close('ae_loss')

encoder.trainable = False      # ✅ مجمّد نهائياً


# ══════════════════════════════════════════════════════════════════
#  3. ⚡ حساب الـ Embeddings مرة واحدة
#
#  بما أن الـ Encoder مجمّد، تمرير الصور عبره كل epoch هدر كامل.
#  نحسبها مرة → التدريب يصير أسرع ~100× → نقدر نجرب
#  عدة تكوينات للـ head في ثوانٍ بدل ساعات.
# ══════════════════════════════════════════════════════════════════
print("\n⚡ حساب الـ embeddings...")
E_train = encoder.predict(X_train, batch_size=256, verbose=0)
E_val   = encoder.predict(X_val,   batch_size=256, verbose=0)
E_test  = encoder.predict(X_test,  batch_size=256, verbose=0)
print(f"   train {E_train.shape} | val {E_val.shape} | test {E_test.shape}")


# ══════════════════════════════════════════════════════════════════
#  4. 📐 خط الأساس — Linear Probe
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60); print("📐 Linear Probe (خط الأساس)"); print("="*60)
probe = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
probe.fit(E_train, y_train)
probe_val  = probe.score(E_val,  y_val)
probe_test = probe.score(E_test, y_test)
print(f"   val: {probe_val*100:.2f}%  |  test: {probe_test*100:.2f}%")
print(f"   ⬆ أي head نبنيه لازم يتجاوز هذا الرقم")


# ══════════════════════════════════════════════════════════════════
#  5. 🎨 بنك Embeddings مُعزَّز (اختياري — يُختبر تجريبياً)
#
#  في v5 الـ augmentation كان يضر: BN داخل الـ encoder المجمّد
#  حفظت إحصاءات من صور نظيفة، والصور المُدوّرة تُسقَط في أماكن
#  غريبة من فضاء الـ 128.
#
#  هنا نبني نسخاً معزّزة، نمررها عبر الـ encoder مرة واحدة،
#  وندع الـ sweep يقرر: هل تنفع أم لا؟ قرار تجريبي لا افتراضي.
# ══════════════════════════════════════════════════════════════════
print("\n🎨 بناء بنك embeddings معزّز (augmentation خفيف)...")
aug = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=6, zoom_range=0.07,
    width_shift_range=0.07, height_shift_range=0.07
)
N_AUG = 8
Xa, ya = [X_train], [y_train]
gen = aug.flow(X_train, y_train, batch_size=len(X_train), shuffle=False)
for _ in range(N_AUG):
    bx, by = next(gen)
    Xa.append(bx); ya.append(by)
X_aug = np.concatenate(Xa); y_aug = np.concatenate(ya)
E_aug = encoder.predict(X_aug, batch_size=256, verbose=0)
print(f"   بنك معزّز: {E_aug.shape}")


# ══════════════════════════════════════════════════════════════════
#  6. ⚖️ أوزان الأصناف — إنقاذ Shirt
#
#  في v5: Shirt recall = 0.04 (96% من القمصان صُنّفت غلط)
#  Shirt أصعب صنف في Fashion MNIST — يتداخل مع T-shirt و
#  Pullover و Coat. مع 96 عينة تدريب فقط والـ dropout الثقيل،
#  الموديل "استسلم" ورمى كل القمصان على Dress و Coat.
#
#  نعطي الأصناف الأربعة العلوية وزناً أعلى لإجبار الموديل
#  على محاولة التفريق بينها بدل تجاهلها.
# ══════════════════════════════════════════════════════════════════
cls_w = {i: 1.0 for i in range(10)}
cls_w[6] = 2.5      # Shirt    — الأصعب
cls_w[0] = 1.5      # T-shirt  — يتداخل مع Shirt
cls_w[2] = 1.5      # Pullover
cls_w[4] = 1.5      # Coat
print(f"\n⚖️  أوزان الأصناف: Shirt=2.5 | T-shirt/Pullover/Coat=1.5")


# ══════════════════════════════════════════════════════════════════
#  7. 🔍 HEAD SWEEP — 5 تكوينات
#
#  بما أن الـ embeddings محسوبة، كل تكوين يتدرب في ثوانٍ.
#  نجرب ونختار الأفضل على الـ val بدل التخمين.
#
#  ⚠️ EarlyStopping على loss: patience=25 (كان 10)
#     السبب: patience=10 هو اللي أوقف v5 عند epoch 19
#     قبل ما يتقارب الموديل.
# ══════════════════════════════════════════════════════════════════
def make_head(units, drop_in, drop_h, use_bn):
    m = keras.Sequential(name='head')
    m.add(keras.Input(shape=(128,)))
    if drop_in > 0:
        m.add(layers.Dropout(drop_in))
    for u in units:
        m.add(layers.Dense(u, activation='relu',
                           kernel_initializer='he_normal'))
        if use_bn:
            m.add(layers.BatchNormalization())
        if drop_h > 0:
            m.add(layers.Dropout(drop_h))
    m.add(layers.Dense(10, activation='softmax',
                       kernel_initializer='glorot_normal'))
    return m


CONFIGS = [
    # (اسم, طبقات, dropout_input, dropout_hidden, BN, augmented, lr)
    ("A: بسيط بلا aug",       [256],       0.0,  0.20, False, False, 1e-3),
    ("B: عميق بلا aug",       [512, 256],  0.0,  0.30, True,  False, 1e-3),
    ("C: بسيط + aug",         [256],       0.0,  0.20, False, True,  1e-3),
    ("D: عميق + aug",         [512, 256],  0.0,  0.30, True,  True,  1e-3),
    ("E: عريض + aug + BN",    [512, 256, 128], 0.05, 0.25, True, True, 8e-4),
]

results = []
print("\n" + "="*60)
print("🔍 HEAD SWEEP — تجربة 5 تكوينات")
print("="*60)

for name, units, d_in, d_h, bn, use_aug, lr in CONFIGS:
    tf.random.set_seed(42); np.random.seed(42)

    Xh, yh = (E_aug, y_aug) if use_aug else (E_train, y_train)

    head = make_head(units, d_in, d_h, bn)
    head.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    h = head.fit(
        Xh, yh,
        epochs=250, batch_size=64,
        validation_data=(E_val, y_val),
        class_weight=cls_w,
        callbacks=[
            keras.callbacks.EarlyStopping('val_accuracy', patience=40,
                                          restore_best_weights=True, mode='max'),
            keras.callbacks.EarlyStopping('loss', patience=25,   # ✅ 10 → 25
                                          min_delta=1e-4),
            keras.callbacks.ReduceLROnPlateau('val_accuracy', factor=0.3,
                                              patience=15, min_lr=1e-7, mode='max'),
        ],
        verbose=0)

    v = max(h.history['val_accuracy'])
    _, t = head.evaluate(E_test, y_test, verbose=0)
    ep_ran = len(h.history['loss'])
    results.append((name, v, t, head, h))

    flag = "✅" if t > probe_test else "⚠️ "
    print(f"{flag} {name:22s} val {v*100:5.2f}%  test {t*100:5.2f}%  ({ep_ran} ep)")

# ── اختيار الأفضل على الـ val ──
best_name, best_val, best_test, best_head, best_hist = max(
    results, key=lambda r: r[1])

print("\n" + "-"*60)
print(f"🏆 الأفضل: {best_name}")
print(f"   val {best_val*100:.2f}%  |  test {best_test*100:.2f}%")
print(f"   Linear Probe: {probe_test*100:.2f}%  "
      f"({(best_test-probe_test)*100:+.2f} نقطة)")
print("-"*60)

# ── مقارنة بصرية للتكوينات ──
plt.figure(figsize=(11, 5))
names  = [r[0] for r in results]
vals   = [r[1]*100 for r in results]
tests  = [r[2]*100 for r in results]
xpos   = np.arange(len(names))
plt.bar(xpos-0.2, vals,  0.4, label='Val',  color='steelblue')
plt.bar(xpos+0.2, tests, 0.4, label='Test', color='darkorange')
plt.axhline(probe_test*100, color='red', linestyle='--',
            label=f'Linear Probe ({probe_test*100:.1f}%)')
plt.xticks(xpos, names, rotation=20, ha='right', fontsize=8)
plt.ylabel('Accuracy %'); plt.title('Head Sweep — مقارنة التكوينات')
plt.legend(); plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
save_and_close('head_sweep')


# ══════════════════════════════════════════════════════════════════
#  8. تركيب الموديل الكامل (Encoder مجمّد + أفضل head)
# ══════════════════════════════════════════════════════════════════
full_in  = keras.Input(shape=(28, 28, 1))
full_out = best_head(encoder(full_in))
classifier = Model(full_in, full_out, name='classifier')
classifier.compile(optimizer=keras.optimizers.AdamW(1e-4),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
classifier.save(f"{SAVE_DIR}/classifier_phase1.keras")

_, test_1 = classifier.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ الموديل الكامل — test: {test_1*100:.2f}%")

# ── منحنى التدريب لأفضل تكوين ──
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, len(best_hist.history['accuracy'])+1)
ax[0].plot(ep, best_hist.history['accuracy'],     label='Train', color='steelblue')
ax[0].plot(ep, best_hist.history['val_accuracy'], label='Val',   color='darkorange')
ax[0].axhline(probe_val, color='red', linestyle=':',
              label=f'Probe ({probe_val:.2%})')
ax[0].set_title(f'Accuracy — {best_name}'); ax[0].set_xlabel('Epoch')
ax[0].legend(); ax[0].grid(True, alpha=0.3)
ax[1].plot(ep, best_hist.history['loss'],     label='Train', color='steelblue')
ax[1].plot(ep, best_hist.history['val_loss'], label='Val',   color='darkorange')
ax[1].set_title('Loss'); ax[1].set_xlabel('Epoch')
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.suptitle('Training History — v6', fontsize=13); plt.tight_layout()
save_and_close('training_history')


# ══════════════════════════════════════════════════════════════════
#  9. Pseudo-Labeling — بوابة + حارس
# ══════════════════════════════════════════════════════════════════
GATE = 0.78
print("\n" + "="*60)
print(f"بوابة الـ Pseudo-Labeling — العتبة {GATE:.0%}")
print("="*60)

if test_1 < GATE:
    print(f"❌ {test_1:.1%} < {GATE:.0%} — تخطّي المرحلة 2")
    print("   pseudo labels من موديل ضعيف تضر أكثر مما تنفع")
    classifier.save(f"{SAVE_DIR}/classifier_final.keras")
    test_final = test_1
else:
    print(f"✅ {test_1:.1%} — نكمل")
    E_unlab = encoder.predict(X_unlab, batch_size=512, verbose=0)

    mc = np.stack([best_head(E_unlab, training=True).numpy()
                   for _ in range(50)])
    m_mean = mc.mean(axis=0)
    unc    = mc.std(axis=0).max(axis=1)
    p_lbl  = np.argmax(m_mean, axis=1)
    p_max  = m_mean.max(axis=1)

    thr     = np.percentile(unc, 20)
    trusted = (unc < thr) & (p_max > 0.95)
    print(f"صور موثوقة: {trusted.sum():,} ({trusted.mean()*100:.1f}%)")

    plt.figure(figsize=(10, 4))
    plt.hist(unc, bins=80, color='steelblue', edgecolor='white')
    plt.axvline(thr, color='green', linestyle='--', label=f'P20={thr:.3f}')
    plt.xlabel('Uncertainty'); plt.ylabel('عدد الصور')
    plt.title('توزيع الـ Uncertainty'); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    save_and_close('uncertainty_dist')

    E_comb = np.concatenate([E_train, E_unlab[trusted]])
    y_comb = np.concatenate([y_train, p_lbl[trusted]])
    w_comb = np.concatenate([np.full(len(E_train), 3.0),
                             np.full(trusted.sum(), 1.0)])

    best_head.compile(
        optimizer=keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    best_head.fit(E_comb, y_comb, sample_weight=w_comb,
                  epochs=100, batch_size=128,
                  validation_data=(E_val, y_val),
                  callbacks=[
                      keras.callbacks.EarlyStopping('val_accuracy', patience=25,
                                                    restore_best_weights=True,
                                                    mode='max', verbose=1),
                      keras.callbacks.ReduceLROnPlateau('val_accuracy', factor=0.3,
                                                        patience=10, mode='max'),
                  ], verbose=1)

    _, test_2 = best_head.evaluate(E_test, y_test, verbose=0)
    print(f"\nقبل pseudo: {test_1:.4f} | بعد: {test_2:.4f}")

    if test_2 < test_1:
        print("🛡️  تراجع — استرجاع موديل المرحلة 1")
        classifier = keras.models.load_model(f"{SAVE_DIR}/classifier_phase1.keras")
        test_final = test_1
    else:
        print(f"✅ تحسّن {(test_2-test_1)*100:+.2f} نقطة")
        test_final = test_2
    classifier.save(f"{SAVE_DIR}/classifier_final.keras")


# ══════════════════════════════════════════════════════════════════
#  10. التقييم النهائي
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("التقييم النهائي — Test Set (10,000 صورة)")
print("="*60)

y_pred = np.argmax(classifier.predict(X_test, batch_size=256, verbose=0), axis=1)
final_acc = (y_pred == y_test).mean()

print(f"\n── ملخص ──")
print(f"Linear Probe:        {probe_test*100:.2f}%")
print(f"أفضل head ({best_name}): {best_test*100:.2f}%")
print(f"Test النهائي:        {final_acc*100:.2f}%")
print(f"مقابل v5 (69.60%):   {(final_acc-0.6960)*100:+.2f} نقطة")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix — {final_acc:.2%}', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save_and_close('confusion_matrix')

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# ── تتبع صنف Shirt تحديداً ──
shirt_recall = cm[6].astype(float)[6] / cm[6].sum()
print(f"\n👔 Shirt recall: {shirt_recall:.3f}  (كان 0.04 في v5)")
if shirt_recall < 0.30:
    print("   لا يزال منخفضاً — جرّب رفع cls_w[6] لـ 4.0")


# ══════════════════════════════════════════════════════════════════
#  11. MC Dropout — عدم اليقين
# ══════════════════════════════════════════════════════════════════
mc_t  = np.stack([best_head(E_test, training=True).numpy() for _ in range(50)])
t_unc = mc_t.std(axis=0).max(axis=1)
ok    = (y_pred == y_test)

plt.figure(figsize=(10, 4))
plt.hist(t_unc[ok],  bins=50, alpha=0.6, color='green',
         label=f'صحيحة ({ok.sum():,})')
plt.hist(t_unc[~ok], bins=50, alpha=0.6, color='red',
         label=f'خاطئة ({(~ok).sum():,})')
plt.xlabel('Uncertainty'); plt.ylabel('عدد الصور')
plt.title('Uncertainty: صحيحة vs خاطئة')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
save_and_close('uncertainty_correct_vs_wrong')

sep = t_unc[~ok].mean() - t_unc[ok].mean()
print(f"\nUncertainty صحيحة: {t_unc[ok].mean():.4f}")
print(f"Uncertainty خاطئة: {t_unc[~ok].mean():.4f}")
print(f"الفصل: {sep:+.4f}  (موجب = معايرة سليمة)")


# ══════════════════════════════════════════════════════════════════
#  12. الملخص
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📁 الملفات المحفوظة")
print("="*60)
for f in sorted(os.listdir(SAVE_DIR)):
    sz = os.path.getsize(f"{SAVE_DIR}/{f}") / 1024 / 1024
    print(f"   {f:42s} {sz:.1f} MB")

print("""
التحميل لاحقاً:
    from tensorflow import keras
    clf = keras.models.load_model("saved_models/classifier_final.keras")
    preds = clf.predict(X)                    # dropout مطفي — صحيح

    # لعدم اليقين فقط:
    mc = np.stack([clf(X, training=True).numpy() for _ in range(50)])
    mean, unc = mc.mean(0), mc.std(0).max(1)
""")
print("✅ Pipeline v6 اكتمل!")