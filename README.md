# Stroke Detection

Makine Öğrenmesi dersi proje ödevi. Bir kişinin sağlık ve yaşam tarzı bilgilerinden geçmişte inme
geçirip geçirmediğini tahmin eden sınıflandırma modelleri kuruyor, farklı modelleri karşılaştırıyor ve
modellerin kararlarını açıklanabilir yapay zeka ile yorumluyoruz.

## Neden inme?

İnme tek bir nedene bağlanamayan bir hastalık. Tansiyon, kolesterol, diyabet, kalp hastalığı, sigara,
yaş gibi risk faktörleri biliniyor; ama bunların bir kişide nasıl bir araya geldiği ve hangisinin ne
kadar ağırlık taşıdığı kişiden kişiye değişiyor. Yaşlı ve tansiyonu yüksek herkes inme geçirmiyor,
belirgin bir risk faktörü olmayan biri geçirebiliyor.

Bu yüzden şu soruyu soruyoruz: **Rutin olarak bilinen değerlerin birlikte örüntüsünden, bir kişinin
inme riski hakkında bir çıkarım yapabilir miyiz ve model bu çıkarımı hangi değerlere dayandırıyor?**

Problemin iki zorluğu projenin yönünü belirliyor:

- **Kaçırmak pahalı.** İnme vakaları veride azınlıkta. Herkese "inme yok" diyen bir model yüksek
  doğruluk alır ama hiçbir vakayı yakalamaz. Bu yüzden başarıyı doğrulukla değil, vakaları
  ne kadar yakaladığımızla ve bunun kaç yanlış alarma mal olduğuyla ölçüyoruz.
- **Gerekçesiz karar tıpta işe yaramaz.** "Risklisin" diyen bir modelin nedenini de söyleyebilmesi
  gerekiyor. Açıklanabilir yapay zeka bu projede bir ek değil, sorunun yarısı.
