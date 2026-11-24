# Decision Tree ve Random Forest: sonuçlar ve final çalıştırma notu

Bu dosya ekip içi kayıt. Eğitilmiş modeller (`models/*.joblib`) repoya konmaz; final çalıştırmada tüm modeller tek bilgisayarda, aynı ortamda yeniden eğitilir. Burada o çalıştırmada kullanılacak en iyi parametreler, denenen bütün değerler ve bu bilgisayarda alınan sonuçlar var. Final çalıştırmanın sonuçları bunlarla karşılaştırılabilir.

- Notebook'lar: `notebooks/07_model_dt.ipynb`, `notebooks/07_model_rf.ipynb`
- Kayıt tarihi: 2026-11-24, branch `dt-rf-tuning-evidence` (main `9cdf282` üstünde)
- Denenen her adayın tam sonucu: `figures/dt_grid_results.csv` (432 aday), `figures/rf_grid_results.csv` (392 aday)

## Ortak ayarlar

| Ayar | Değer |
|---|---|
| Veri | `data/train.csv` / `data/test.csv` (01'in çıktısı, %80/%20 katmanlı, `RANDOM_STATE=42`) |
| CV | 5 katlı stratified, `shared_cv()` |
| Seçim metriği | CV PR-AUC (fold ortalaması) |
| Std | 5 katın standart sapması, ddof=1 (`cv_evaluate` ile aynı) |
| Eşik | CV OOF skorlarından, %80 recall'u en az yanlış alarmla sağlayan eşik (bölüm 6) |
| Rastgele tahmin seviyesi | PR-AUC ≈ 0,087 (eğitimdeki inme oranı) |

## Final çalıştırmada kullanılacak parametreler

```python
# Decision Tree
DecisionTreeClassifier(max_depth=4, min_samples_leaf=80, criterion="gini",
                       class_weight=None, random_state=RANDOM_STATE)
build_pipeline(model, balancing="none")

# Random Forest
RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=40,
                       max_features="sqrt", class_weight=None,
                       random_state=RANDOM_STATE, n_jobs=-1)
build_pipeline(model, balancing="none")
```

Notebook'lar GridSearch'ü kendisi yapıp en iyiyi seçiyor; final çalıştırmada notebook olduğu gibi çalıştırılırsa aynı parametreler çıkmalı. Çıkmazsa (farklı sklearn sürümü vb.) bu dosyadaki değerlerle karşılaştırılır.

**Dengeleme yok seçildi.** İki modelde de "dengeleme yok" en yüksek CV PR-AUC'yi verdi, SMOTE en düşüğü. Bu yüzden iki model 0,5 eşikte hiç inme demiyor (aşağıdaki tablolarda recall/precision/F1 0,5 eşikte 0). Dengesizlik eşikle çözülüyor: çalışma eşiği bölüm 6'dan gelir. 09 da her model için kendi %80 recall eşiğini seçiyor.

## Decision Tree

### Denenen değerler

| Parametre | Değerler |
|---|---|
| dengeleme | yok, SMOTE, class_weight="balanced" |
| max_depth | 2, 3, 4, 5, 6, 8, 10, None |
| min_samples_leaf | 1, 5, 10, 20, 40, 80, 120, 160, 240 |
| criterion | gini, entropy |

3 × 8 × 9 × 2 = 432 aday. Süre bu bilgisayarda ~26 sn.

### Baseline ve ayarlı model (CV PR-AUC, ortalama ± std)

| Model | PR-AUC |
|---|---|
| Varsayılan, dengeleme yok | 0,095 ± 0,010 |
| Varsayılan + SMOTE | 0,102 ± 0,013 |
| **Ayarlı (yok, derinlik 4, leaf 80, gini)** | **0,170 ± 0,010** (ROC-AUC 0,689, 12 yaprak) |

### Parametre başına etki (diğerleri en iyi değerde sabit)

Dengeleme (recall/precision/F1 0,5 eşikte):

| dengeleme | CV PR-AUC | std | recall | precision | F1 | train PR-AUC | train − CV |
|---|---|---|---|---|---|---|---|
| **yok** | **0,170** | 0,010 | 0,000 | 0,000 | 0,000 | 0,208 | 0,038 |
| class_weight (balanced) | 0,162 | 0,018 | 0,650 | 0,138 | 0,228 | 0,209 | 0,047 |
| smote | 0,133 | 0,011 | 0,516 | 0,146 | 0,227 | 0,167 | 0,034 |

max_depth:

| max_depth | CV PR-AUC | std | train PR-AUC | train − CV |
|---|---|---|---|---|
| 2 | 0,155 | 0,014 | 0,167 | 0,012 |
| 3 | 0,163 | 0,014 | 0,190 | 0,027 |
| **4** | **0,170** | 0,010 | 0,208 | 0,038 |
| 5 | 0,166 | 0,013 | 0,220 | 0,054 |
| 6 | 0,167 | 0,015 | 0,226 | 0,059 |
| 8 | 0,168 | 0,014 | 0,228 | 0,060 |
| 10 | 0,168 | 0,015 | 0,228 | 0,060 |
| None | 0,168 | 0,015 | 0,228 | 0,060 |

min_samples_leaf:

| min_samples_leaf | CV PR-AUC | std | train PR-AUC | train − CV |
|---|---|---|---|---|
| 1 | 0,158 | 0,014 | 0,270 | 0,113 |
| 5 | 0,162 | 0,016 | 0,259 | 0,097 |
| 10 | 0,155 | 0,013 | 0,249 | 0,094 |
| 20 | 0,144 | 0,010 | 0,233 | 0,090 |
| 40 | 0,164 | 0,016 | 0,217 | 0,052 |
| **80** | **0,170** | 0,010 | 0,208 | 0,038 |
| 120 | 0,168 | 0,013 | 0,202 | 0,034 |
| 160 | 0,161 | 0,012 | 0,196 | 0,035 |
| 240 | 0,154 | 0,009 | 0,177 | 0,023 |

criterion: **gini 0,170** ± 0,010, entropy 0,157 ± 0,010.

**Yorum:** 432 adayın 74'ü en iyinin bir std içinde; derinlik 4'ten sonrası CV'de bir şey katmıyor, yalnız train-CV farkını büyütüyor. Farklar küçük olduğu için daha sade ağaç (derinlik 4) seçildi. Önceki seçim (SMOTE, derinlik 5, leaf 80, entropy) 0,168 ± 0,045 idi.

### Test (bu bilgisayarda, eşik 0,053)

| Recall | Precision | F1 | ROC-AUC | PR-AUC | Yakalanan | Yanlış alarm |
|---|---|---|---|---|---|---|
| 0,909 | 0,135 | 0,236 | 0,742 | 0,208 | 70 / 77 | 447 / 804 |

CV OOF (havuzlanmış) PR-AUC: 0,177.

## Random Forest

### Denenen değerler

| Parametre | Değerler |
|---|---|
| dengeleme | yok, SMOTE, class_weight="balanced", class_weight="balanced_subsample" |
| max_depth | 2, 3, 4, 6, 8, 12, None |
| min_samples_leaf | 1, 5, 10, 20, 40, 80, 160 |
| max_features | "sqrt", 0.5 |
| n_estimators | 300 (sabit, aranmadı) |

4 × 7 × 7 × 2 = 392 aday. Grid süresi bu bilgisayarda ~4 dk, notebook tamamı ~6 dk.

### Baseline ve ayarlı model (CV PR-AUC, ortalama ± std)

| Model | PR-AUC |
|---|---|
| Varsayılan (100 ağaç), dengeleme yok | 0,169 ± 0,015 |
| Varsayılan + SMOTE | 0,165 ± 0,020 |
| **Ayarlı (yok, derinlik 3, leaf 40, sqrt, 300 ağaç)** | **0,221 ± 0,033** (ROC-AUC 0,712) |

### Parametre başına etki (diğerleri en iyi değerde sabit)

Dengeleme (recall/precision/F1 0,5 eşikte):

| dengeleme | CV PR-AUC | std | recall | precision | F1 | train PR-AUC | train − CV |
|---|---|---|---|---|---|---|---|
| **yok** | **0,221** | 0,033 | 0,000 | 0,000 | 0,000 | 0,242 | 0,020 |
| class_weight (balanced_subsample) | 0,213 | 0,030 | 0,572 | 0,152 | 0,240 | 0,249 | 0,036 |
| class_weight (balanced) | 0,212 | 0,033 | 0,595 | 0,149 | 0,238 | 0,245 | 0,033 |
| smote | 0,195 | 0,026 | 0,556 | 0,152 | 0,239 | 0,240 | 0,045 |

max_depth:

| max_depth | CV PR-AUC | std | train PR-AUC | train − CV |
|---|---|---|---|---|
| 2 | 0,216 | 0,030 | 0,230 | 0,015 |
| **3** | **0,221** | 0,033 | 0,242 | 0,020 |
| 4 | 0,208 | 0,022 | 0,264 | 0,056 |
| 6 | 0,211 | 0,020 | 0,297 | 0,087 |
| 8 | 0,203 | 0,022 | 0,310 | 0,107 |
| 12 | 0,206 | 0,019 | 0,313 | 0,107 |
| None | 0,206 | 0,019 | 0,313 | 0,107 |

min_samples_leaf:

| min_samples_leaf | CV PR-AUC | std | train PR-AUC | train − CV |
|---|---|---|---|---|
| 1 | 0,210 | 0,026 | 0,291 | 0,081 |
| 5 | 0,209 | 0,027 | 0,277 | 0,068 |
| 10 | 0,214 | 0,026 | 0,270 | 0,056 |
| 20 | 0,216 | 0,028 | 0,258 | 0,043 |
| **40** | **0,221** | 0,033 | 0,242 | 0,020 |
| 80 | 0,209 | 0,024 | 0,230 | 0,022 |
| 160 | 0,197 | 0,016 | 0,216 | 0,019 |

max_features: **sqrt 0,221** ± 0,033, 0.5 0,211 ± 0,019.

**Yorum:** 392 adayın 316'sı en iyinin bir std içinde; parametreler arası farkların çoğu gürültü düzeyinde. Net olanlar: ayarlı model varsayılanı geçiyor, SMOTE geride, derinlik arttıkça train-CV farkı 0,015'ten 0,107'ye çıkıyor (ezber). Önceki seçim balanced_subsample, derinlik 4, leaf 10, sqrt idi.

### Test (bu bilgisayarda, eşik 0,068)

| Recall | Precision | F1 | ROC-AUC | PR-AUC | Yakalanan | Yanlış alarm |
|---|---|---|---|---|---|---|
| 0,870 | 0,151 | 0,258 | 0,766 | 0,237 | 67 / 77 | 376 / 804 |

CV OOF (havuzlanmış) PR-AUC: 0,190.

## Final çalıştırma için kontrol

1. `01_data_preprocessing` önce çalışır, `data/train.csv` ve `data/test.csv` üretir (aynı seed → aynı bölme).
2. `07_model_dt` ve `07_model_rf` baştan sona çalışır; seçilen parametreler yukarıdakilerle aynı mı bakılır.
3. CV ve test sayıları yukarıdakilerle karşılaştırılır; küçük farklar (sürüm, işlemci) beklenir, seçim değişirse not edilir.
4. Sonra `09_model_comparison` çalışır.
