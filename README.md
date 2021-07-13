# cat-vs-dogs-classifier

# bne08
Bu proje, girdi görüntüsünü bir köpek veya kedi görüntüsü olarak sınıflandırmayı amaçlamaktadır. 
Sisteme verdiğiniz görüntü girişi analiz edilecek ve tahmin edilen sonuç çıktı olarak verilecektir.
Görüntüyü sınıflandırmak için makine öğrenmesi algoritması [Konvolüsyonel Sinir Ağları] kullanılır.Projemizi direk indirerek python editorlerinizde çalıştırabilirsiniz.
Train veri setimizin çalışmasıyla ağımızı eğiterek epoch değerlerini görebilirsiniz ve test setimizden alınan örneklerle karşılaştırma yapabilir sistemi test edebilirsiniz.

Projede kullanılan metaryeller :
-Ubuntu
-Sublimetext 
-Terminal 
Kullanılan Kütüphaneler :
-OPENCV : OpenCV (Open Source Computer Vision) açık kaynak kodlu görüntü işleme kütüphanesidir. 
-NUMPY : NumPy, bilimsel/matematiksel/mantıksal/istatistiksel hesaplama için oluşturulmuş bir python kütüphanesidir. 
-TENSOR FLOW : Google’ın çıkarttığı bir makine öğrenmesi kütüphanesidir. 
-PANDAS : Pandas python programlama dili için yüksek performanslı , kullanımı kolay veri analiz yapıları  ve araçları sağlayan açık kaynaklı bir BSD lisanslı kütüphanedir. 

METHOTLAR
CNN  (Konvolüsyonel Sinir Ağları)
•	CNN, belge tanıma  ve görüntü tanıma gibi görevlerde oldukça iyi performans sağlayan bir tür derin mimaridir.
•	Nesne tespitleri, yüzleri tanıma vb. CNN ‘lerin yaygın olarak kullanıldığı alanlardan bazılarıdır.
CNN görüntüyü çeşitli katmanlarda işler 5 katmandan olusur. 
1.	Konvolüsyonel Katman: Resmin özellikleri saptamak için kullanılır. Resme bazı  filtreler uygulanır kenar belirleme , görüntü yumuşatma,görüntü  keskinleştirme filtreleri vb 
2.Düzleştirilmiş Doğrusal Birim Katmanı(Rectified Linear UnitsLayer(ReLu)) :
Bu katman konvolüsyon katmanlarından sonra gelir ve CNN nöronlarının çıktıları için en yaygın şekilde devreye sokulan doğrultucu birim olarak bilinir
2.Düzleştirilmiş Doğrusal Birim Katmanı(Rectified Linear UnitsLayer(ReLu))
Bu katman konvolüsyon katmanlarından sonra gelir ve CNN nöronlarının çıktıları için en yaygın şekilde devreye sokulan doğrultucu birim olarak bilinir
3.Havuzlama Katmanı (Pooling Layer) :
Havuzlama genellikle ReLu katmanından sonra yerleştirilir. Temel amacı, sonraki konvolüsyon katmanı için giriş boyutunu (Genişlik x Yükseklik) azaltmaktır. 

