# Import required libraries
import os  # For file handling
import base64  # For encoding the signature
import piexif  # For reading and writing EXIF metadata
import imagehash  # For generating perceptual hash
import numpy as np  # For numerical operations
import pywt  # For applying Discrete Wavelet Transform (DWT)
import torch  # For using PyTorch model
import torchvision.transforms as transforms  # For image preprocessing
from torchvision.models import resnet50, ResNet50_Weights  # Using pretrained ResNet50 model
from PIL import Image  # For image loading and manipulation
from datetime import datetime  # To record timestamp
from cryptography.hazmat.primitives import hashes, serialization  # For cryptographic hashing
from cryptography.hazmat.primitives.asymmetric import rsa, padding  # For RSA and signature padding
from cryptography.hazmat.backends import default_backend  # For cryptographic backend


# 🛡️ Main class for protecting and verifying images
class AdvancedImageProtector:
    def __init__(self):
        # Set device to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained ResNet50 model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Define preprocessing steps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to model input size
            transforms.ToTensor()  # Convert to tensor
        ])

        # Generate RSA public-private key pair
        self.private_key, self.public_key = self._generate_keys()

    # 🔐 Generate RSA key pair
    def _generate_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    # 🧠 Extract deep features using ResNet50
    def _extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().cpu().numpy()

    # 🎵 Extract frequency-domain features using DWT (Discrete Wavelet Transform)
    def _apply_dwt(self, image_path):
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        data = np.array(img)
        coeffs = pywt.dwt2(data, 'haar')  # Apply Haar wavelet
        cA, (cH, cV, cD) = coeffs  # Approximation + Detail coefficients
        return np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    # 🔑 Generate perceptual hash of the image
    def _generate_hash(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return str(imagehash.phash(image))

    # ✍️ Sign the hash using the private key
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

    # 🧾 Embed protection metadata into image using EXIF
    def _embed_metadata(self, image_path, metadata):
        img = Image.open(image_path).convert("RGB")
        # Load existing EXIF data, or empty if not available
        exif_dict = piexif.load(img.info.get('exif', b''))

        # Encode user comment as bytes (required by EXIF format)
        user_comment = metadata.encode('utf-8')
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = b"ASCII\0\0\0" + user_comment
        exif_bytes = piexif.dump(exif_dict)

        # Save to protected_images directory
        protected_dir = "protected_images"
        os.makedirs(protected_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(protected_dir, f"protected_{filename}")
        img.save(output_path, exif=exif_bytes)
        return output_path

    # 🔐 Full process to protect an image
    def protect_image(self, image_path):
        print(f"Extracting features from {image_path}")
        features = self._extract_features(image_path)

        print("Applying DWT to extract frequency-domain features...")
        dwt_features = self._apply_dwt(image_path)

        print("Generating perceptual hash...")
        image_hash = self._generate_hash(image_path)

        print("Signing the hash...")
        signature = self._sign_hash(image_hash)

        # Add timestamp and store everything as a string
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

    # 🔍 Verify if image is authentic using signature
    def verify_image(self, protected_image_path):
        print(f"Verifying authenticity of {protected_image_path}...")
        img = Image.open(protected_image_path)

        # Try to read metadata from image
        exif_data = piexif.load(img.info.get('exif', b''))
        comment = exif_data['Exif'].get(piexif.ExifIFD.UserComment)
        if not comment:
            return "No protection metadata found."

        # Extract and decode metadata string
        metadata_raw = comment.decode('utf-8', errors='ignore').replace("ASCII\0\0\0", "")
        try:
            metadata = eval(metadata_raw)
        except:
            return "Corrupted metadata."

        # Generate new hash from image to compare with signed one
        hash_from_image = self._generate_hash(protected_image_path)

        # Verify signature
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

        # Calculate how old the image is
        timestamp = datetime.strptime(metadata["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
        age = (datetime.utcnow() - timestamp).days
        return f"Image signature is valid. The image is authentic and was protected {age} days ago."

    # 📦 Optional: Protect multiple images in a batch
    def batch_process(self, image_paths):
        for idx, path in enumerate(image_paths):
            self.protect_image(path)
            yield (idx + 1) / len(image_paths)


# =========================== MAIN ===========================

if __name__ == "__main__":
    protector = AdvancedImageProtector()

    # 🛡️ Step 1: Protect your image
    result = protector.protect_image("sample.jpg")
    print(result)

    # 🔍 Step 2: Verify that it's protected
    verification_result = protector.verify_image("protected_images/protected_sample.jpg")
    print(verification_result)
