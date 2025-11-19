import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class SegmentationModel:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Definisi Label dan Warna (GUI juga pake ini)
        self.VOC_LABELS = {
            0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
            5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
            10: "cow", 11: "dining table", 12: "dog", 13: "horse", 14: "motorbike",
            15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train",
            20: "tv/monitor"
        }

        # Palet warna yang lebih kontras untuk segmentasi objek
        self.VOC_COLORS = np.array([
            [  0,   0,   0], # 0=background (hitam transparan)
            [128,   0,   0], # 1=aeroplane (merah tua)
            [  0, 128,   0], # 2=bicycle (hijau tua)
            [128, 128,   0], # 3=bird (kuning tua)
            [  0,   0, 128], # 4=boat (biru tua)
            [128,   0, 128], # 5=bottle (ungu tua)
            [  0, 128, 128], # 6=bus (teal tua)
            [192, 192, 192], # 7=car (abu-abu terang)
            [255, 128,   0], # 8=cat (orange terang)
            [128,  64,   0], # 9=chair (coklat)
            [ 64, 128,   0], # 10=cow (hijau kecoklatan)
            [192,   0, 128], # 11=dining table (merah muda keunguan)
            [128,   0, 255], # 12=dog (ungu kebiruan)
            [  0, 192, 128], # 13=horse (cyan)
            [192, 128,   0], # 14=motorbike (orange kecoklatan)
            [255,   0,   0], # 15=person (merah terang)
            [  0, 255,   0], # 16=potted plant (hijau terang)
            [  0,   0, 255], # 17=sheep (biru terang)
            [255, 255,   0], # 18=sofa (kuning terang)
            [255,   0, 255], # 19=train (magenta)
            [  0, 255, 255]  # 20=tv/monitor (cyan terang)
        ], dtype=np.uint8)


    def load_model(self):
        """Memuat model DeepLabV3 Pretrained"""
        print(f"Memuat model menggunakan device: {self.device}...")
        self.model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.model.to(self.device)
        self.model.eval()
        print("Model berhasil dimuat.")

    def predict(self, original_image):
        """
        Menerima PIL Image, melakukan inferensi, 
        dan mengembalikan gambar hasil overlay (PIL Image).
        """
        if self.model is None:
            raise Exception("Model belum dimuat.")

        input_tensor = self.preprocess(original_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()

        segmented_mask = self._decode_segmap(output_predictions)
        
        segmented_mask = segmented_mask.resize(original_image.size, resample=Image.NEAREST)
        overlay_image = Image.blend(original_image.convert("RGB"), segmented_mask, alpha=0.6)
        
        return overlay_image, output_predictions

    def _decode_segmap(self, image_mask_array):
        """Mengubah matriks kelas menjadi gambar berwarna RGB (Internal Method)"""
        r = np.zeros_like(image_mask_array).astype(np.uint8)
        g = np.zeros_like(image_mask_array).astype(np.uint8)
        b = np.zeros_like(image_mask_array).astype(np.uint8)

        for l in range(0, len(self.VOC_LABELS)):
            idx = image_mask_array == l
            r[idx] = self.VOC_COLORS[l, 0]
            g[idx] = self.VOC_COLORS[l, 1]
            b[idx] = self.VOC_COLORS[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return Image.fromarray(rgb)