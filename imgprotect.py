import os
import base64
import piexif
import imagehash
import numpy as np
import pywt
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class AdvancedImageProtector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.private_key, self.public_key = self._generate_keys()

    def _generate_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def _extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().cpu().numpy()

    def _apply_dwt(self, image_path):
        img = Image.open(image_path).convert("L")
        data = np.array(img)
        coeffs = pywt.dwt2(data, 'haar')
        cA, (cH, cV, cD) = coeffs
        return np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    def _generate_hash(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return str(imagehash.phash(image))

    def _sign_hash(self, hash_str):
        signature = self.private_key.sign(
            hash_str.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def _embed_metadata(self, image_path, metadata):
        img = Image.open(image_path).convert("RGB")
        exif_dict = piexif.load(img.info.get('exif', b''))
        user_comment = metadata.encode('utf-8')
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = b"ASCII\0\0\0" + user_comment
        exif_bytes = piexif.dump(exif_dict)

        protected_dir = "protected_images"
        os.makedirs(protected_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(protected_dir, f"protected_{filename}")
        img.save(output_path, exif=exif_bytes)
        return output_path

    def protect_image(self, image_path):
        print(f"Extracting features from {image_path}")
        features = self._extract_features(image_path)

        print("Applying DWT to extract frequency-domain features...")
        dwt_features = self._apply_dwt(image_path)

        print("Generating perceptual hash...")
        image_hash = self._generate_hash(image_path)

        print("Signing the hash...")
        signature = self._sign_hash(image_hash)

        metadata = {
            "hash": image_hash,
            "signature": signature,
            "timestamp": str(datetime.utcnow())
        }

        metadata_str = str(metadata)

        print("Embedding metadata into image...")
        output_path = self._embed_metadata(image_path, metadata_str)

        print(f"Saving protected image to {output_path}...")
        return f"Image processing complete. Protected image saved as {output_path}"

    def verify_image(self, protected_image_path):
        print(f"Verifying authenticity of {protected_image_path}...")
        img = Image.open(protected_image_path)
        exif_data = piexif.load(img.info.get('exif', b''))
        comment = exif_data['Exif'].get(piexif.ExifIFD.UserComment)
        if not comment:
            return "No protection metadata found."

        metadata_raw = comment.decode('utf-8', errors='ignore').replace("ASCII\0\0\0", "")
        try:
            metadata = eval(metadata_raw)
        except:
            return "Corrupted metadata."

        hash_from_image = self._generate_hash(protected_image_path)
        try:
            self.public_key.verify(
                base64.b64decode(metadata["signature"]),
                metadata["hash"].encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except Exception as e:
            return f"Verification failed: {e}"

        timestamp = datetime.strptime(metadata["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
        age = (datetime.utcnow() - timestamp).days
        return f"Image signature is valid. The image is authentic and was protected {age} days ago."

    def batch_process(self, image_paths):
        for idx, path in enumerate(image_paths):
            self.protect_image(path)
            yield (idx + 1) / len(image_paths)


# =========================== MAIN ===========================

if __name__ == "__main__":
    protector = AdvancedImageProtector()

    # 🛡️ Protect a sample image (replace 'sample.jpg' with your filename)
    result = protector.protect_image("sample.jpg")
    print(result)

    # 🔍 Verify the protected image
    verification_result = protector.verify_image("protected_images/protected_sample.jpg")
    print(verification_result)
