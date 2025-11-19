import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import csv

from unet_model import UNet


class SegmentationModel:
    def __init__(self, model_path='unet_best.pth', class_dict_path='labels_class_dict.csv'):
        self.model = None
        self.model_path = model_path
        self.class_dict_path = class_dict_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size =  128
        
        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load class dictionary from CSV
        self.VOC_LABELS, self.VOC_COLORS = self._load_class_dict()

    def _load_class_dict(self):
        """Load class names and colors from CSV file"""
        labels = {}
        colors = []
        
        if not os.path.exists(self.class_dict_path):
            raise FileNotFoundError(f"Class dictionary not found: {self.class_dict_path}")
        
        with open(self.class_dict_path, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                labels[idx] = row['class_names']
                colors.append([int(row['r']), int(row['g']), int(row['b'])])
        
        return labels, np.array(colors, dtype=np.uint8)

    def load_model(self):
        """Load trained U-Net model"""
        print(f"Memuat model U-Net dari {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file tidak ditemukan: {self.model_path}\n"
                f"Pastikan file unet_best.pth ada di folder yang sama dengan script ini."
            )
        
        # Initialize model with correct number of classes
        num_classes = len(self.VOC_LABELS)
        self.model = UNet(n_channels=3, n_classes=num_classes)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model berhasil dimuat pada device: {self.device}")
        if 'val_accuracy' in checkpoint:
            print(f"Model accuracy: {checkpoint['val_accuracy']:.4f}")

    def predict(self, original_image):
        """
        Menerima PIL Image, melakukan inferensi, 
        dan mengembalikan gambar hasil overlay (PIL Image) dan predictions array.
        """
        if self.model is None:
            raise Exception("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        # Simpan ukuran asli
        original_size = original_image.size
        
        # Preprocess
        input_tensor = self.preprocess(original_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Get predictions
        output_predictions = output[0].argmax(0).byte().cpu().numpy()

        # Decode segmentation map to colored image
        segmented_mask = self._decode_segmap(output_predictions)
        
        # Resize mask back to original size
        segmented_mask = segmented_mask.resize(original_size, resample=Image.NEAREST)
        
        # Blend with original image
        overlay_image = Image.blend(original_image.convert("RGB"), segmented_mask, alpha=0.6)
        
        return overlay_image, output_predictions

    def _decode_segmap(self, image_mask_array):
        """Mengubah matriks kelas menjadi gambar berwarna RGB"""
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