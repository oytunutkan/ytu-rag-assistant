# ingest.py
# YTÜ mevzuat dosyalarını ChromaDB'ye aktarır.
import os
import glob
import shutil
import re

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from category_rules import CATEGORY_RULES

# App ve test runner tarafında da aynı path kullanılmalı.
CHROMA_PATH = "chroma_db"
# Chroma içindeki koleksiyon adı. Aynı DB içinde farklı koleksiyonlar tutulabilir.
COLLECTION_NAME = "ytu_mevzuat"

# Word belgelerini vektöre çevirecek embedding modeli.
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"

# Her kategori için farklı chunk ayarı kullanıyoruz.
# chunk_size: metnin kaç karakterlik parçalara bölüneceği.
# chunk_overlap: komşu parçaların ne kadar ortak metin taşıyacağı.
# Küçük ve maddesel yönergelerde chunk küçük; şartların birlikte görülmesi gereken konularda daha büyük tutuldu.
CATEGORY_CHUNK_CONFIG = {
    # Kısa ve belirgin maddeli kategoriler: küçük/orta chunk yeterli.
    "bitirme_calismasi":     {"chunk_size": 500,  "chunk_overlap": 75},
    "sinav_itiraz":          {"chunk_size": 500,  "chunk_overlap": 50},
    "yuzde_on":              {"chunk_size": 500,  "chunk_overlap": 75},
    "akademik_danismanlik":  {"chunk_size": 500,  "chunk_overlap": 75},

    # Mazeret gibi detay maddelerinde küçük chunk doğru hükmü yakalamayı kolaylaştırır.
    "mazeret_sinavi":        {"chunk_size": 450,  "chunk_overlap": 100},
    "mezuniyet_sinavi":      {"chunk_size": 650,  "chunk_overlap": 100},
    "sinav_yonergesi":       {"chunk_size": 750,  "chunk_overlap": 100},


    "staj":                  {"chunk_size": 500,  "chunk_overlap": 75},
    "ders_kayit":            {"chunk_size": 650,  "chunk_overlap": 100},
    "yaz_okulu":             {"chunk_size": 750,  "chunk_overlap": 100},
    "cap":                   {"chunk_size": 900,  "chunk_overlap": 150},
    "yandal":                {"chunk_size": 500,  "chunk_overlap": 75},
    "azami_sure":            {"chunk_size": 650,  "chunk_overlap": 120},
    "esdegerlik_intibak":    {"chunk_size": 750,  "chunk_overlap": 120},


    "ozel_ogrenci_ytuye_gelen":  {"chunk_size": 750, "chunk_overlap": 120},
    "ozel_ogrenci_ytuden_giden": {"chunk_size": 750, "chunk_overlap": 120},

    "lisansustu_genel_basvuru":                   {"chunk_size": 800, "chunk_overlap": 150},
    "lisansustu_tezli_yuksek_lisans":             {"chunk_size": 900, "chunk_overlap": 150},
    "lisansustu_tezsiz_yuksek_lisans":            {"chunk_size": 850, "chunk_overlap": 150},
    "lisansustu_doktora":                         {"chunk_size": 900, "chunk_overlap": 150},
    "lisansustu_sanatta_yeterlik":                {"chunk_size": 900, "chunk_overlap": 150},
    "lisansustu_bilimsel_hazirlik_kayit_ogretim": {"chunk_size": 850, "chunk_overlap": 150},
    "lisansustu_kontenjan_danismanlik_ek_sure":   {"chunk_size": 750, "chunk_overlap": 120},

    "diploma_bilgileri":                 {"chunk_size": 600, "chunk_overlap": 100},
    "diploma_mezuniyet_tarihleri":       {"chunk_size": 400, "chunk_overlap": 75},
    "diploma_eki_yandal_belgeler":       {"chunk_size": 550, "chunk_overlap": 100},
    "diploma_teslim_kayip_ikinci_nusha": {"chunk_size": 650, "chunk_overlap": 120},

    "kurum_ici_yatay_gecis":            {"chunk_size": 1000, "chunk_overlap": 150},
    "kurumlar_arasi_yatay_gecis":       {"chunk_size": 1000, "chunk_overlap": 150},
    "merkezi_yerlestirme_yatay_gecis":  {"chunk_size": 900,  "chunk_overlap": 150},
    "yurt_disi_yatay_gecis":            {"chunk_size": 1000, "chunk_overlap": 150},
    "dikey_gecis":                      {"chunk_size": 850,  "chunk_overlap": 120},
    "yatay_gecis_genel":                {"chunk_size": 900,  "chunk_overlap": 150},
    "isletmede_mesleki_egitim":         {"chunk_size": 1000, "chunk_overlap": 150},

    # V8: İşletmede Mesleki Eğitim yönergesi 5 Word dosyasına bölündü.
    # Her alt başlık kendi kategori metadata'sıyla Chroma'ya yazılacak.
    "isletmede_mesleki_egitim_tanimlar":                     {"chunk_size": 650, "chunk_overlap": 100},
    "isletmede_mesleki_egitim_komisyonlar_gorevler":         {"chunk_size": 750, "chunk_overlap": 120},
    "isletmede_mesleki_egitim_isletme_egitici_sorumlu_ogrenci": {"chunk_size": 750, "chunk_overlap": 120},
    "isletmede_mesleki_egitim_basvuru_degerlendirme_itiraz": {"chunk_size": 800, "chunk_overlap": 120},
    "isletmede_mesleki_egitim_diger_hukumler":               {"chunk_size": 750, "chunk_overlap": 120},

    "onceki_ogrenme":                   {"chunk_size": 500,  "chunk_overlap": 100},

    # Hiçbir kategoriye düşmeyen dosyalar için güvenli varsayılan ayar.
    "genel":                 {"chunk_size": 600,  "chunk_overlap": 75},
}

# Mikro madde/süre/istisna sorularını daha iyi yakalamak için güncel chunk ayarları.
# Çok küçük chunk, madde koşullarını bölüyordu; çok büyük chunk da semantik gürültü yaratıyordu.
# Bu ara değerlerle madde + alt bentlerin aynı chunk içinde kalması hedeflenir.
V12_CHUNK_OVERRIDES = {
    "mazeret_sinavi":        {"chunk_size": 650,  "chunk_overlap": 150},
    "sinav_yonergesi":       {"chunk_size": 650,  "chunk_overlap": 130},
    "staj":                  {"chunk_size": 720,  "chunk_overlap": 150},
    "ders_kayit":            {"chunk_size": 800,  "chunk_overlap": 160},
    "onceki_ogrenme":        {"chunk_size": 700,  "chunk_overlap": 150},
    "ozel_ogrenci_ytuye_gelen":  {"chunk_size": 850, "chunk_overlap": 170},
    "ozel_ogrenci_ytuden_giden": {"chunk_size": 850, "chunk_overlap": 170},
    "esdegerlik_intibak":    {"chunk_size": 850,  "chunk_overlap": 160},
    "diploma_bilgileri":                 {"chunk_size": 720, "chunk_overlap": 140},
    "diploma_mezuniyet_tarihleri":       {"chunk_size": 520, "chunk_overlap": 100},
    "diploma_eki_yandal_belgeler":       {"chunk_size": 700, "chunk_overlap": 140},
    "diploma_teslim_kayip_ikinci_nusha": {"chunk_size": 760, "chunk_overlap": 150},
    "kurum_ici_yatay_gecis":            {"chunk_size": 1050, "chunk_overlap": 180},
    "kurumlar_arasi_yatay_gecis":       {"chunk_size": 1050, "chunk_overlap": 180},
    "merkezi_yerlestirme_yatay_gecis":  {"chunk_size": 980,  "chunk_overlap": 170},
    "yurt_disi_yatay_gecis":            {"chunk_size": 1050, "chunk_overlap": 180},
    "lisansustu_tezli_yuksek_lisans":             {"chunk_size": 950, "chunk_overlap": 180},
    "lisansustu_doktora":                         {"chunk_size": 980, "chunk_overlap": 180},
    "lisansustu_bilimsel_hazirlik_kayit_ogretim": {"chunk_size": 900, "chunk_overlap": 170},
    "lisansustu_kontenjan_danismanlik_ek_sure":   {"chunk_size": 850, "chunk_overlap": 160},
}
CATEGORY_CHUNK_CONFIG.update(V12_CHUNK_OVERRIDES)

# Mikro kartlar atomik olduğu için küçük-orta chunk yeterli;
# komşu kartların bağlamı kaybolmasın diye overlap korunur.
FINAL_CHUNK_OVERRIDES = {
    "staj": {"chunk_size": 620, "chunk_overlap": 130},
    "yandal": {"chunk_size": 620, "chunk_overlap": 130},
    "cap": {"chunk_size": 800, "chunk_overlap": 160},
    "onceki_ogrenme": {"chunk_size": 680, "chunk_overlap": 150},
    "ozel_ogrenci_ytuye_gelen": {"chunk_size": 760, "chunk_overlap": 150},
    "ozel_ogrenci_ytuden_giden": {"chunk_size": 760, "chunk_overlap": 150},
    "sinav_yonergesi": {"chunk_size": 680, "chunk_overlap": 140},
    "isletmede_mesleki_egitim_tanimlar": {"chunk_size": 650, "chunk_overlap": 130},
    "isletmede_mesleki_egitim_komisyonlar_gorevler": {"chunk_size": 650, "chunk_overlap": 130},
    "isletmede_mesleki_egitim_isletme_egitici_sorumlu_ogrenci": {"chunk_size": 650, "chunk_overlap": 130},
    "isletmede_mesleki_egitim_basvuru_degerlendirme_itiraz": {"chunk_size": 720, "chunk_overlap": 150},
    "ders_kayit": {"chunk_size": 700, "chunk_overlap": 150},
    "yaz_okulu": {"chunk_size": 700, "chunk_overlap": 140},
}
CATEGORY_CHUNK_CONFIG.update(FINAL_CHUNK_OVERRIDES)


def normalize_text(text: str) -> str:
    # Dosya adı ve anahtar kelime karşılaştırmalarında Türkçe karakter farkı sorun olmasın diye metni sadeleştiriyoruz.
    return (
        str(text).lower()
        .replace("ı", "i").replace("İ", "i")
        .replace("ğ", "g").replace("ü", "u")
        .replace("ş", "s").replace("ö", "o")
        .replace("ç", "c")
        .replace("’", "'").replace("‘", "'").replace("`", "'")
    )


def detect_category_from_filename(file_path: str) -> str:
    # Dosya adından kategori bulur. Örneğin "Tezsiz_Yuksek_Lisans" geçen dosya ilgili lisansüstü kategorisine düşer.
    normalized_name = normalize_text(os.path.basename(file_path))

    # category_rules.py içindeki "files" anahtarları dosya adında aranır.
    # Böylece ingest ve app tarafı aynı kategori isimlerini kullanır.
    candidates = []
    for category, rule in CATEGORY_RULES.items():
        for file_key in rule.get("files", []):
            candidates.append((category, file_key))

    # Daha uzun/özel dosya anahtarları önce denenir. Böylece genel kelimeler özel kategorilerin önüne geçmez.
    candidates.sort(key=lambda item: len(normalize_text(item[1])), reverse=True)

    for category, file_key in candidates:
        if normalize_text(file_key) in normalized_name:
            return category

    return "genel"


def add_metadata(doc, dosya_yolu: str, kategori: str):
    # Her dokümana kaynak, kategori ve kullanılan chunk ayarlarını metadata olarak ekliyoruz.
    # App tarafındaki filtreli arama bu "kategori" alanına göre çalışıyor.
    config = CATEGORY_CHUNK_CONFIG.get(kategori, CATEGORY_CHUNK_CONFIG["genel"])

    if "mikro_kart" in normalize_text(dosya_yolu) or "micro_card" in normalize_text(dosya_yolu):
        doc.metadata["kaynak_tipi"] = "mikro_kart"
        doc.metadata["guvenilirlik_puani"] = 12
    else:
        doc.metadata["kaynak_tipi"] = "resmi_yonetmelik"
        doc.metadata["guvenilirlik_puani"] = 10
    doc.metadata["kategori"] = kategori
    doc.metadata["dosya_adi"] = os.path.basename(dosya_yolu)
    doc.metadata["chunk_size_config"] = config["chunk_size"]
    doc.metadata["chunk_overlap_config"] = config["chunk_overlap"]


def extract_madde_label(text: str) -> str:
    """Chunk içinde görünen ilk MADDE/Madde başlığını kısa metadata/prefix olarak çıkarır."""
    if not text:
        return ""
    match = re.search(r"(?im)^\s*(MADDE|Madde)\s+\d+[^\n]{0,120}", text)
    if match:
        return " ".join(match.group(0).split())
    return ""


def enrich_chunk_for_retrieval(chunk, kategori: str):
    """V12: Retrieval için chunk başına kısa kaynak/kategori/madde etiketi ekler.
    Bu, özellikle 'kim karar verir', 'kaç gün', 'hangi madde' gibi mikro sorularda
    embedding'in doğru bağlamı yakalamasını kolaylaştırır.
    """
    dosya_adi = chunk.metadata.get("dosya_adi", "")
    madde_label = extract_madde_label(chunk.page_content)
    if madde_label:
        chunk.metadata["madde_etiketi"] = madde_label

    prefix_parts = [
        f"Kaynak dosya: {dosya_adi}",
        f"Kategori: {kategori}",
    ]
    if chunk.metadata.get("kaynak_tipi") == "mikro_kart":
        prefix_parts.append("Mikro Kart: Evet")
    if madde_label:
        prefix_parts.append(f"Madde etiketi: {madde_label}")

    prefix = "\n".join(prefix_parts) + "\n---\n"
    if not chunk.page_content.startswith("Kaynak dosya:"):
        chunk.page_content = prefix + chunk.page_content


def main():
    # Ingest işleminin ana akışı burada başlar: dosyaları oku, parçala, embed et, Chroma’ya yaz.
    print(f"Çalışma klasörü: {os.getcwd()}", flush=True)

    if not os.path.exists("data"):
        raise FileNotFoundError("data klasörü bulunamadı.")

    # data klasöründeki Word + mikro kart TXT dosyalarını alıyoruz.
    # Eski veya yeni mikro kart TXT klasörleri data altında kalabilir.
    docx_dosyalari = glob.glob("data/**/*.docx", recursive=True)
    txt_dosyalari = glob.glob("data/**/*.txt", recursive=True)
    dokuman_dosyalari = docx_dosyalari + txt_dosyalari

    print(f"Bulunan Word dosyası sayısı: {len(docx_dosyalari)}", flush=True)
    print(f"Bulunan TXT mikro kart sayısı: {len(txt_dosyalari)}", flush=True)

    if not dokuman_dosyalari:
        raise FileNotFoundError("data klasöründe .docx veya .txt dosyası yok.")

    # Dosyaları kategoriye göre gruplayacağız; çünkü her kategori farklı chunk ayarı kullanıyor.
    dokumanlar_by_kategori = {}

    for dosya_yolu in dokuman_dosyalari:
        # Word veya TXT dosyasını düz metin dokümanı olarak yüklüyoruz.
        if dosya_yolu.lower().endswith(".docx"):
            loader = Docx2txtLoader(dosya_yolu)
        elif dosya_yolu.lower().endswith(".txt"):
            loader = TextLoader(dosya_yolu, encoding="utf-8")
        else:
            continue

        yuklenen_dokumanlar = loader.load()
        # Dosya adındaki anahtar kelimelere göre kategori belirlenir.
        kategori = detect_category_from_filename(dosya_yolu)

        for doc in yuklenen_dokumanlar:
            add_metadata(doc, dosya_yolu, kategori)

        dokumanlar_by_kategori.setdefault(kategori, []).extend(yuklenen_dokumanlar)
        print(f"Okundu: {dosya_yolu} | Kategori: {kategori}", flush=True)

    print("\nSSS sayfası şimdilik atlandı.", flush=True)
    print("\nMetinler kategoriye göre parçalanıyor...", flush=True)

    tum_chunklar = []

    for kategori, dokumanlar in dokumanlar_by_kategori.items():
        # Her kategori kendi chunk_size / overlap ayarıyla parçalanır.
        config = CATEGORY_CHUNK_CONFIG.get(kategori, CATEGORY_CHUNK_CONFIG["genel"])

        # RecursiveCharacterTextSplitter önce paragraf/satır gibi doğal ayrımları dener, gerekirse kelime/karakter seviyesine iner.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            # Ayırıcılar önemlidir: önce paragraf, sonra satır, sonra cümle ve kelime sınırları denenir.
            separators=["\nMADDE ", "\nMadde ", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

        # Seçilen ayarlara göre dokümanlar küçük arama parçalarına ayrılır.
        chunklar = splitter.split_documents(dokumanlar)

        for chunk in chunklar:
            # Chunk metadata’sı tekrar garanti altına alınır; filtreli retrieval bu bilgiye güveniyor.
            chunk.metadata["kategori"] = kategori
            chunk.metadata["chunk_size_config"] = config["chunk_size"]
            chunk.metadata["chunk_overlap_config"] = config["chunk_overlap"]
            enrich_chunk_for_retrieval(chunk, kategori)

        tum_chunklar.extend(chunklar)

        print(
            f"  [{kategori}] chunk_size={config['chunk_size']} | "
            f"overlap={config['chunk_overlap']} | "
            f"{len(dokumanlar)} doküman → {len(chunklar)} chunk",
            flush=True,
        )

    print(f"\nToplam kategori sayısı: {len(dokumanlar_by_kategori)}", flush=True)
    print(f"Toplam chunk sayısı: {len(tum_chunklar)}", flush=True)

    if not tum_chunklar:
        raise ValueError("Hiç chunk oluşmadı.")

    # Yeni ingest temiz başlasın diye aynı path’te eski Chroma varsa siliyoruz.
    print("\nEski ChromaDB kontrol ediliyor...", flush=True)
    if os.path.exists(CHROMA_PATH):
        print(f"Eski ChromaDB siliniyor: {CHROMA_PATH}", flush=True)
        shutil.rmtree(CHROMA_PATH)

    # Embedding modeli her chunk’ı sayısal vektöre çevirir.
    print("\nEmbedding modeli yükleniyor...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Chroma, metin chunklarını ve embedding vektörlerini birlikte saklayan vektör veritabanımızdır.
    print("ChromaDB oluşturuluyor...", flush=True)
    db = Chroma.from_documents(
        documents=tum_chunklar,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    kayit_sayisi = db._collection.count()
    print(f"Chroma kayıt sayısı: {kayit_sayisi}", flush=True)

    if kayit_sayisi == 0:
        raise ValueError("ChromaDB oluşturuldu ama kayıt sayısı 0.")

    print(f"\nVeritabanı başarıyla '{CHROMA_PATH}' klasörüne kaydedildi.", flush=True)
    print("Test runner ve app tarafında aynı CHROMA_PATH değerini kullanmayı unutma.", flush=True)


if __name__ == "__main__":
    # Dosya doğrudan çalıştırılırsa ingest işlemini başlatır.
    main()
