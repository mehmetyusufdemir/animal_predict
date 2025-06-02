import customtkinter as ctk  # CustomTkinter'ı ctk olarak import ediyoruz
from tkinter import filedialog  # Dosya seçme diyaloğu için
from PIL import Image  # Görüntü işlemleri için
import torch
import torchvision.transforms as transforms
import timm

# Arayüz ayarları (açık tema, koyu tema veya sistem teması)
ctk.set_appearance_mode("System")  # "System", "Dark", "Light" olabilir
ctk.set_default_color_theme("blue")  # Temalar: "blue", "green", "dark-blue"

# --- Model ve Sınıf İsimleri (Bu kısımları daha önce tanımlamıştık) ---
MODEL_PATH = 'best_animal_classifier_deit.pth'  # Model dosyasının yolu
NUM_CLASSES = 90  # Sınıf sayısı (doğruluğundan emin ol)
# Sınıf isimlerinin bir listesi gerekiyor. Eğer bir dosyadan okuyacaksanız veya manuel girecekseniz:
# Örnek: class_names = ["antelope", "badger", ..., "zebra"]
# Şimdilik geçici bir class_names listesi oluşturalım, bunu gerçek listenizle değiştirmeniz gerekecek.
# Daha önceki notebook'taki train_dataset.classes'dan alabilirsiniz.
# Eğer Jupyter Notebook'tan bu class_names listesini kopyalayamıyorsanız,
# geçici olarak kısa bir liste kullanıp daha sonra güncelleyebilirsiniz.
# En sağlıklısı, train_dataset oluşturulduğunda bu listeyi bir .txt dosyasına kaydetmek
# ve arayüzde bu dosyadan okumaktır.

# GEÇİCİ - GERÇEK class_names LİSTENİZLE DEĞİŞTİRİN!
# Örnek: Bu listeyi bir önceki veri yükleme adımlarındaki 'train_dataset.classes' ile doldurun.
# class_names = ['sınıf1', 'sınıf2', ..., 'sınıf90']
# Eğer class_names'i almanın bir yolu yoksa, tahmin sonucunda sadece sınıf index'ini gösterebiliriz.
# Şimdilik bu kısmı boş bırakıyorum, tahmin fonksiyonunda buna değineceğiz.
class_names = []  # BU LİSTEYİ GERÇEK VERİLERLE DOLDURUN!
# Eğer hemen dolduramıyorsanız, sınıf isimlerini almak için bir yöntem bulana kadar
# tahmin fonksiyonu sadece sınıf index'ini ve olasılığı döndürebilir.
# Örnek olarak, bir önceki notebook'tan class_names = train_dataset.classes satırındaki listeyi buraya kopyalayabilirsiniz.
# Örneğin:
# class_names = ['antelope', 'badger', 'bat', ..., 'zebra'] # 90 elemanlı tam liste olmalı.


# Model yükleme fonksiyonu (uygulama başlarken bir kez çağrılabilir)
loaded_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_animal_model():
    global loaded_model
    try:
        model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()  # Değerlendirme moduna al
        loaded_model = model
        print(f"Model '{MODEL_PATH}' başarıyla yüklendi ve '{device}' cihazına taşındı.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        loaded_model = None


# Görüntü ön işleme dönüşümleri (doğrulama setindekine benzer)
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Direkt 224x224'e boyutlandırıyoruz
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Tahmin fonksiyonu
def predict_image(image_path):
    if loaded_model is None:
        result_label.configure(text="Model yüklenemedi!")
        return

    try:
        image = Image.open(image_path).convert("RGB")

        # Görüntüyü arayüzde göstermek için (boyutlandırılmış)
        display_image = image.resize((250, 250))
        img_ctk = ctk.CTkImage(light_image=display_image, dark_image=display_image, size=(250, 250))
        image_label.configure(image=img_ctk)
        image_label.image = img_ctk  # Referansı tutmak için

        # Model için ön işleme
        img_tensor = eval_transforms(image)
        img_tensor = img_tensor.unsqueeze(0)  # Batch boyutu ekle [C, H, W] -> [1, C, H, W]
        img_tensor = img_tensor.to(device)

        with torch.no_grad():  # Gradyan hesaplaması yapma
            outputs = loaded_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]  # Olasılıkları al
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class_idx = predicted_idx.item()
        confidence_score = confidence.item()

        if class_names and 0 <= predicted_class_idx < len(class_names):
            predicted_animal = class_names[predicted_class_idx]
            result_label.configure(text=f"Tahmin: {predicted_animal}\nGüven: {confidence_score:.2%}")
        else:
            # Eğer class_names listesi boşsa veya index hatalıysa
            result_label.configure(
                text=f"Tahmin Edilen Sınıf Index'i: {predicted_class_idx}\nGüven: {confidence_score:.2%}")
            if not class_names:
                print("UYARI: class_names listesi boş. Sadece index gösteriliyor.")


    except Exception as e:
        result_label.configure(text=f"Tahmin hatası: {e}")
        print(f"Tahmin sırasında hata: {e}")


# Arayüz elemanları için fonksiyonlar
def select_image():
    file_path = filedialog.askopenfilename(title="Bir hayvan resmi seçin",
                                           filetypes=(("JPEG dosyaları", "*.jpg *.jpeg"),
                                                      ("PNG dosyaları", "*.png"),
                                                      ("Tüm Dosyalar", "*.*")))
    if file_path:
        print(f"Seçilen dosya: {file_path}")
        # Resmi göster ve tahmin yap
        predict_image(file_path)
    else:
        print("Dosya seçilmedi.")


# Ana uygulama sınıfı veya direkt pencere oluşturma
app = ctk.CTk()
app.title("Hayvan Sınıflandırma Uygulaması")
app.geometry("600x550")  # Pencere boyutunu biraz büyüttüm

# Frame'ler (isteğe bağlı, düzen için)
main_frame = ctk.CTkFrame(master=app)
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Başlık Etiketi
title_label = ctk.CTkLabel(master=main_frame, text="Hayvan Türü Tahmini", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=10)

# Resim Seçme Butonu
select_button = ctk.CTkButton(master=main_frame, text="Resim Seç", command=select_image, width=200, height=40)
select_button.pack(pady=10)

# Resim Gösterme Alanı (başlangıçta boş)
image_label = ctk.CTkLabel(master=main_frame, text="Lütfen bir resim seçin.", width=250, height=250)
image_label.pack(pady=10)

# Sonuç Etiketi
result_label = ctk.CTkLabel(master=main_frame, text="Tahmin sonucu burada gösterilecek.", font=ctk.CTkFont(size=16))
result_label.pack(pady=20)

# Uygulama başladığında modeli yükle
load_animal_model()

app.mainloop()