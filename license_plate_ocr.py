import cv2
import numpy as np
import easyocr
import re
from typing import List, Tuple, Optional, Dict
import torch

class LicensePlateOCR:
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = True):
        """
        Initialize EasyOCR for license plate recognition
        
        Args:
            languages: List of languages to recognize
            use_gpu: Whether to use GPU for processing
        """
        # Check GPU availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        print(f"Initializing EasyOCR with GPU: {self.use_gpu}")
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=self.use_gpu)
        
        # License plate patterns for different countries
        # Add more patterns as needed
        self.plate_patterns = {
            'romania': r'^[A-Z]{1,2}\s*\d{2,3}\s*[A-Z]{3}$',  # Romanian format
            'generic': r'^[A-Z0-9\s\-]{4,12}$',  # Generic alphanumeric
            'eu': r'^[A-Z]{1,3}\s*[A-Z0-9]{1,4}\s*[A-Z0-9]{1,4}$',  # European formats
            'us': r'^[A-Z0-9]{1,7}$'  # US formats
        }
        
        # Common character confusions in license plates
        self.char_substitutions = {
            'O': '0', '0': 'O',  # O and 0 confusion
            'I': '1', '1': 'I',  # I and 1 confusion
            'S': '5', '5': 'S',  # S and 5 confusion
            'B': '8', '8': 'B',  # B and 8 confusion
            'G': '6', '6': 'G',  # G and 6 confusion
            'Z': '2', '2': 'Z'   # Z and 2 confusion
        }
        
    def preprocess_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR results
        
        Args:
            plate_image: Input license plate image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
            
        # Resize if too small
        height, width = gray.shape
        if width < 200:
            scale = 200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=30)
        
        # Apply threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Edge preservation filter
        filtered = cv2.bilateralFilter(cleaned, 11, 17, 17)
        
        return filtered
        
    def read_plate(self, plate_image: np.ndarray, preprocess: bool = True) -> Dict:
        """
        Read license plate text from image
        
        Args:
            plate_image: License plate image
            preprocess: Whether to preprocess the image
            
        Returns:
            Dictionary with OCR results
        """
        # Preprocess image if requested
        if preprocess:
            processed_image = self.preprocess_plate_image(plate_image)
        else:
            processed_image = plate_image
            
        # Run OCR
        results = self.reader.readtext(processed_image)
        
        # Extract text and confidence
        all_text = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            # Clean and format text
            cleaned_text = self.clean_plate_text(text)
            if cleaned_text:
                all_text.append(cleaned_text)
                confidences.append(confidence)
                
        # Combine text
        if all_text:
            combined_text = ' '.join(all_text)
            avg_confidence = np.mean(confidences)
            
            # Try to match against known patterns
            matched_format = self.match_plate_format(combined_text)
            
            # Apply error correction
            corrected_text = self.apply_error_correction(combined_text)
            
            return {
                'text': corrected_text,
                'raw_text': combined_text,
                'confidence': avg_confidence,
                'format': matched_format,
                'valid': matched_format is not None
            }
        else:
            return {
                'text': '',
                'raw_text': '',
                'confidence': 0.0,
                'format': None,
                'valid': False
            }
            
    def clean_plate_text(self, text: str) -> str:
        """
        Clean and normalize license plate text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Convert to uppercase
        text = text.upper()
        
        # Remove special characters except hyphen and space
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
        
    def match_plate_format(self, text: str) -> Optional[str]:
        """
        Match plate text against known formats
        
        Args:
            text: Cleaned plate text
            
        Returns:
            Matched format name or None
        """
        # Remove spaces for pattern matching
        text_no_spaces = text.replace(' ', '')
        
        for format_name, pattern in self.plate_patterns.items():
            if re.match(pattern, text) or re.match(pattern, text_no_spaces):
                return format_name
                
        return None
        
    def apply_error_correction(self, text: str) -> str:
        """
        Apply common error corrections for license plates
        
        Args:
            text: Plate text
            
        Returns:
            Corrected text
        """
        # Split into parts
        parts = text.split()
        corrected_parts = []
        
        for part in parts:
            # Check if part should be letters or numbers based on position
            # This is a simplified approach - adjust based on your region's format
            
            # Try character substitutions for ambiguous characters
            corrected_part = part
            
            # You can implement more sophisticated error correction here
            # based on your specific license plate formats
            
            corrected_parts.append(corrected_part)
            
        return ' '.join(corrected_parts)
        
    def process_multiple_angles(self, plate_images: List[np.ndarray]) -> Dict:
        """
        Process multiple images of the same plate for better accuracy
        
        Args:
            plate_images: List of plate images from different angles/frames
            
        Returns:
            Best OCR result
        """
        results = []
        
        for image in plate_images:
            result = self.read_plate(image)
            if result['valid']:
                results.append(result)
                
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'format': None,
                'valid': False
            }
            
        # Find most common text
        texts = [r['text'] for r in results]
        text_counts = {}
        
        for text in texts:
            text_counts[text] = text_counts.get(text, 0) + 1
            
        # Get most frequent text
        best_text = max(text_counts, key=text_counts.get)
        
        # Calculate average confidence for best text
        best_confidences = [r['confidence'] for r in results if r['text'] == best_text]
        avg_confidence = np.mean(best_confidences)
        
        # Get format
        best_format = next(r['format'] for r in results if r['text'] == best_text)
        
        return {
            'text': best_text,
            'confidence': avg_confidence,
            'format': best_format,
            'valid': True,
            'num_reads': len(plate_images),
            'consensus_count': text_counts[best_text]
        }
        
    def benchmark_ocr_settings(self, plate_image: np.ndarray) -> Dict:
        """
        Test different preprocessing settings to find best OCR results
        
        Args:
            plate_image: License plate image
            
        Returns:
            Dictionary with results from different settings
        """
        results = {}
        
        # Test without preprocessing
        results['no_preprocessing'] = self.read_plate(plate_image, preprocess=False)
        
        # Test with preprocessing
        results['with_preprocessing'] = self.read_plate(plate_image, preprocess=True)
        
        # Test with different threshold values
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
        
        for thresh_val in [100, 127, 150, 180]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            results[f'threshold_{thresh_val}'] = self.read_plate(binary, preprocess=False)
            
        # Find best result
        best_result = max(results.items(), 
                         key=lambda x: x[1]['confidence'] if x[1]['valid'] else 0)
        
        return {
            'best_method': best_result[0],
            'best_result': best_result[1],
            'all_results': results
        }