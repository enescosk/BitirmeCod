# ENS492 / 48k Update

`ens492` klasoru karisik bir CWRU koleksiyonu iceriyor. Icinde:

- `12k` drive-end verileri
- `48k` drive-end verileri
- `12k` fan-end verileri

aynı anda bulunuyor.

Bu yuzden yapilmasi gereken sey, sadece tek bir `12k ball fault` zip'ini islemek degil; ayni feature-extraction pipeline'ini tum bu karisik CWRU klasorune uygulamakti.

Bu guncellemede pipeline su sekilde duzenlendi:

- CWRU dosya ID'lerinden class/output bilgileri otomatik cikariliyor
- `12k` ve `48k` sampling rate otomatik ayarlaniyor
- fan-end dosyalarda `FE_time`, drive-end dosyalarda `DE_time` kanali seciliyor
- her segment icin time, frequency ve time-frequency feature'lari cikariliyor
- her satira output label'lari ekleniyor:
  - `output_fault_status`
  - `output_bearing_type`
  - `output_fault_location`
  - `output_fault_diameter_in`
  - `output_motor_load_hp`
  - `output_outer_race_position`

## Run Result

Config:
- [ens492_config.json](/Users/ec/Desktop/automation/hidayet/BitirmeCod/configs/ens492_config.json)

Outputs:
- [combined_features_with_outputs.csv](/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/ens492_run/combined_features_with_outputs.csv)
- [metadata_outputs.csv](/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/ens492_run/metadata_outputs.csv)
- [time_domain_features.csv](/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/ens492_run/time_domain_features.csv)
- [frequency_domain_features.csv](/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/ens492_run/frequency_domain_features.csv)
- [time_frequency_features.csv](/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/ens492_run/time_frequency_features.csv)
- [run_summary.json](/Users/ec/Desktop/automation/hidayet/BitirmeCod/outputs/ens492_run/run_summary.json)

Summary:
- total segments: `28330`
- `12k` segments: `7388`
- `48k` segments: `20942`
- labels: `healthy`, `inner_race`, `outer_race`, `ball`
- missing values: `0`

Kisacasi, `48k data ne yapacagiz?` sorusunun cevabi su:

`12k` icin yaptigimiz seyin aynisini `48k` ve fan-end dosyalar icin de ayni veritabani mantigiyla yaptik; artik hepsi tek bir CWRU output database icinde birlesmis durumda.
