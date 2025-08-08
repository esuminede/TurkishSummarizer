import zeyrek
analizer = zeyrek.MorphAnalyzer()

for parse in analizer.analyze('benim')[0]:
    print(parse)

### LEMMAZITATION Yapmak HANGİ AŞAMADA GEREKLİ ANLAMAYA ÇALIŞIYORUM
### AMA ZEYREK KÜTÜPHANESİNİN DÜZGÜN BİR DÖKÜMANTASYONU YOK!!!