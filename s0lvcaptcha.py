#!/usr/bin/env python3
"""
s0lvcaptcha - Multi-service CAPTCHA solver with smart consensus
Advanced CAPTCHA solving tool using OCR and external services with intelligent consensus algorithm
Created by: Fabian Peña (stuxboynet)
GitHub: https://github.com/stuxboynet/s0lvcaptcha
Version: 1.0.0
License: BSD 3-Clause
"""

import base64
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
import time
import argparse
import os
import json
from io import BytesIO
from collections import Counter

__version__ = "1.0.0"

class S0lvCaptcha:
    def __init__(self):
        self.services = {}
        self.config_file = 's0lvcaptcha_config.json'
        self.show_banner()
        self.load_saved_config()
    
    def show_banner(self):
        """Display s0lvcaptcha banner"""
        banner = f"""

███████╗ ██████╗ ██╗    ██╗   ██╗ ██████╗ █████╗ ██████╗ ████████╗ ██████╗██╗  ██╗ █████╗ 
██╔════╝██╔═████╗██║    ██║   ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
███████╗██║██╔██║██║    ██║   ██║██║     ███████║██████╔╝   ██║   ██║     ███████║███████║
╚════██║████╔╝██║██║    ╚██╗ ██╔╝██║     ██╔══██║██╔═══╝    ██║   ██║     ██╔══██║██╔══██║
███████║╚██████╔╝███████╗╚████╔╝ ╚██████╗██║  ██║██║        ██║   ╚██████╗██║  ██║██║  ██║
╚══════╝ ╚═════╝ ╚══════╝ ╚═══╝   ╚═════╝╚═╝  ╚═╝╚═╝        ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
                                                                                          
        🚀 Multi-service CAPTCHA solver with smart consensus
        💀 OCR + 2captcha + AntiCaptcha + CapMonster  
        ⚡ Created by Fabian Peña (stuxboynet)
                    Version {__version__}
"""
        print(banner)
        
    def load_saved_config(self):
        """Load saved configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.services = json.load(f)
                print("🔧 EXISTING CONFIGURATION FOUND")
                print("=" * 60)
                self.show_saved_config()
                self.ask_for_each_service()
            except Exception as e:
                print(f"❌ Error reading configuration: {e}")
                self.setup_new_config()
        else:
            print("🔧 FIRST TIME SETUP")
            print("=" * 60)
            self.setup_new_config()
    
    def show_saved_config(self):
        """Show saved configuration"""
        print("📋 Saved APIs:")
        for service, key in self.services.items():
            masked = key[:8] + '...' + key[-8:] if len(key) > 16 else key[:4] + '...'
            print(f"   🔹 {service}: {masked}")
        print()
    
    def ask_for_each_service(self):
        """Ask for each service individually"""
        print("💡 Review each saved API:")
        print()
        
        new_services = {}
        services_info = [
            ('2captcha', '2captcha'),
            ('anticaptcha', 'AntiCaptcha'), 
            ('capmonster', 'CapMonster')
        ]
        
        for service_key, service_name in services_info:
            print(f"🔹 {service_name}:")
            
            if service_key in self.services:
                existing_key = self.services[service_key]
                masked = existing_key[:8] + '...' + existing_key[-8:] if len(existing_key) > 16 else existing_key[:4] + '...'
                print(f"   Saved API: {masked}")
                
                choice = input(f"   Keep this API? (y/n/Enter=skip): ").strip().lower()
                
                if choice in ['y', 'yes', '']:
                    new_services[service_key] = existing_key
                    print("   ✅ API kept")
                elif choice in ['n', 'no']:
                    new_key = input(f"   New API for {service_name}: ").strip()
                    if new_key:
                        new_services[service_key] = new_key
                        print("   ✅ New API configured")
                    else:
                        print("   ⏭️  Skipped")
                else:
                    print("   ⏭️  Skipped")
            else:
                new_key = input(f"   API for {service_name} (Enter=skip): ").strip()
                if new_key:
                    new_services[service_key] = new_key
                    print("   ✅ API configured")
                else:
                    print("   ⏭️  Skipped")
            print()
        
        self.services = new_services
        self.save_config()
    
    def setup_new_config(self):
        """Setup configuration from scratch"""
        print("🔐 Configure your CAPTCHA service APIs:")
        print()
        
        services_info = [
            ('2captcha', '2captcha'),
            ('anticaptcha', 'AntiCaptcha'), 
            ('capmonster', 'CapMonster')
        ]
        
        for service_key, service_name in services_info:
            api_key = input(f"🔹 {service_name} API key (Enter=skip): ").strip()
            if api_key:
                self.services[service_key] = api_key
                print(f"   ✅ {service_name} configured")
            else:
                print(f"   ⏭️  {service_name} skipped")
            print()
        
        self.save_config()
    
    def save_config(self):
        """Save configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.services, f, indent=2)
            print(f"💾 Configuration saved to {self.config_file}")
            print(f"📊 Active services: {len(self.services)}")
            if self.services:
                print(f"   Configured: {', '.join(self.services.keys())}")
            else:
                print("   ⚠️  Only local OCR available")
            print()
            return True
        except Exception as e:
            print(f"❌ Error saving: {e}")
            return False
    
    def manage_config(self):
        """Manage configuration"""
        print("🔧 CONFIGURATION MANAGEMENT")
        print("=" * 40)
        print("1. View current configuration")
        print("2. Change individual API")
        print("3. Reconfigure everything")
        print("4. Delete configuration")
        print("5. Exit")
        
        choice = input("\nOption (1-5): ").strip()
        
        if choice == "1":
            if self.services:
                print("\n📋 Current configuration:")
                for service, key in self.services.items():
                    masked = key[:8] + '...' + key[-8:] if len(key) > 16 else key[:4] + '...'
                    print(f"   🔹 {service}: {masked}")
            else:
                print("\n❌ No APIs configured")
                
        elif choice == "2":
            self.change_individual_api()
            
        elif choice == "3":
            self.services = {}
            self.setup_new_config()
            
        elif choice == "4":
            confirm = input("⚠️  Type 'DELETE' to confirm: ").strip()
            if confirm == 'DELETE':
                try:
                    if os.path.exists(self.config_file):
                        os.remove(self.config_file)
                    self.services = {}
                    print("✅ Configuration deleted")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Cancelled")
                
        elif choice == "5":
            return
        else:
            print("❌ Invalid option")
    
    def change_individual_api(self):
        """Change individual API"""
        if not self.services:
            print("\n❌ No APIs configured")
            return
            
        print("\n🔧 Available APIs:")
        services_list = list(self.services.keys())
        for i, service in enumerate(services_list, 1):
            masked = self.services[service][:8] + '...' + self.services[service][-8:]
            print(f"{i}. {service}: {masked}")
        
        try:
            choice = int(input(f"\nWhich one to change? (1-{len(services_list)}, 0=cancel): ").strip())
            if choice == 0:
                return
            elif 1 <= choice <= len(services_list):
                service = services_list[choice - 1]
                new_key = input(f"New API for {service}: ").strip()
                if new_key:
                    self.services[service] = new_key
                    self.save_config()
                    print(f"✅ {service} updated")
                else:
                    print("❌ Cancelled")
            else:
                print("❌ Invalid number")
        except ValueError:
            print("❌ Invalid input")
    
    def reset_config(self):
        """Reset configuration completely"""
        print("🔄 RESETTING CONFIGURATION...")
        try:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
                print("✅ Configuration file deleted")
            else:
                print("ℹ️  No configuration file found")
            
            self.services = {}
            print("✅ APIs cleared from memory")
            print("🔄 Reset completed")
            
        except Exception as e:
            print(f"❌ Error during reset: {e}")
    
    def preprocess_multiple(self, img):
        """Multiple preprocessing techniques improved for CAPTCHAs"""
        img_array = np.array(img)
        processed_images = []
        
        # Resize if too small
        height, width = img_array.shape[:2] if len(img_array.shape) == 3 else img_array.shape
        if height < 50 or width < 150:
            scale_factor = max(3, 150 // width, 50 // height)
            new_height = height * scale_factor
            new_width = width * scale_factor
            img_resized = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            img_resized = img_array
        
        # Original resized
        processed_images.append(('Original', Image.fromarray(img_resized)))
        
        # Grayscale
        if len(img_resized.shape) == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_resized
        processed_images.append(('Gray', Image.fromarray(gray)))
        
        # Invert if white text on black background
        if np.mean(gray) < 128:
            inverted = cv2.bitwise_not(gray)
            processed_images.append(('Inverted', Image.fromarray(inverted)))
        
        # Multiple binarizations
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('Otsu', Image.fromarray(otsu)))
        
        # Adaptive binarization - Gaussian
        binary_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(('AdaptiveGauss', Image.fromarray(binary_gauss)))
        
        # Adaptive binarization - Mean
        binary_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(('AdaptiveMean', Image.fromarray(binary_mean)))
        
        # Multiple fixed thresholds
        for thresh_val in [120, 140, 160, 180]:
            _, thresh_fixed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            processed_images.append((f'Thresh{thresh_val}', Image.fromarray(thresh_fixed)))
        
        # Noise removal with different kernels
        denoised3 = cv2.medianBlur(gray, 3)
        processed_images.append(('Denoised3', Image.fromarray(denoised3)))
        
        denoised5 = cv2.medianBlur(gray, 5)
        processed_images.append(('Denoised5', Image.fromarray(denoised5)))
        
        # Bilateral filter (preserves edges)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_images.append(('Bilateral', Image.fromarray(bilateral)))
        
        # Multiple morphological operations
        kernel_small = np.ones((1,1), np.uint8)
        kernel_medium = np.ones((2,2), np.uint8)
        kernel_large = np.ones((3,3), np.uint8)
        
        # Erosion and dilation
        eroded_small = cv2.erode(gray, kernel_small, iterations=1)
        dilated_small = cv2.dilate(eroded_small, kernel_small, iterations=1)
        processed_images.append(('Morph_Small', Image.fromarray(dilated_small)))
        
        eroded_medium = cv2.erode(gray, kernel_medium, iterations=1)
        dilated_medium = cv2.dilate(eroded_medium, kernel_medium, iterations=1)
        processed_images.append(('Morph_Medium', Image.fromarray(dilated_medium)))
        
        # Opening (removes small noise)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_small)
        processed_images.append(('Opened', Image.fromarray(opened)))
        
        # Closing (fills holes)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_small)
        processed_images.append(('Closed', Image.fromarray(closed)))
        
        # SPECIAL TECHNIQUES FOR DISTRACTION LINES
        
        # 1. Detection of horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        # Detect vertical lines  
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine detected lines
        lines_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Remove lines from original image
        no_lines = cv2.subtract(gray, lines_mask)
        processed_images.append(('NoLines', Image.fromarray(no_lines)))
        
        # 2. Alternative method: use inpainting to "erase" lines
        try:
            # Create more aggressive line mask
            lines_thick = cv2.dilate(lines_mask, kernel_small, iterations=1)
            inpainted = cv2.inpaint(gray, lines_thick, 3, cv2.INPAINT_TELEA)
            processed_images.append(('Inpainted', Image.fromarray(inpainted)))
        except:
            pass
        
        # 3. Aggressive median filter for thin lines
        median_strong = cv2.medianBlur(gray, 7)
        processed_images.append(('MedianStrong', Image.fromarray(median_strong)))
        
        # 4. Aggressive opening operation to eliminate thin lines
        opening_aggressive = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_medium, iterations=2)
        processed_images.append(('OpeningAggressive', Image.fromarray(opening_aggressive)))
        
        # 5. Combination: remove lines + binarization
        no_lines_otsu = cv2.threshold(no_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed_images.append(('NoLinesOtsu', Image.fromarray(no_lines_otsu)))
        
        # 6. Morphological gradient (highlights edges, reduces lines)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_small)
        processed_images.append(('Gradient', Image.fromarray(gradient)))
        
        # 7. Top-hat (highlights small text)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_medium)
        processed_images.append(('TopHat', Image.fromarray(tophat)))
        
        # Sharpening (enhance edges)
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharp)
        processed_images.append(('Sharpened', Image.fromarray(sharpened)))
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(('Enhanced', Image.fromarray(enhanced)))
        
        # Apply anti-line techniques to enhanced image too
        enhanced_no_lines = cv2.subtract(enhanced, lines_mask)
        processed_images.append(('EnhancedNoLines', Image.fromarray(enhanced_no_lines)))
        
        return processed_images
    
    def solve_with_advanced_ocr(self, image_data):
        """OCR with multiple configurations and preprocessing"""
        results = []
        
        # Decode image
        if image_data.startswith('data:image'):
            base64_data = image_data.split(',')[1]
        else:
            base64_data = image_data
            
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Multiple preprocessing
        processed_images = self.preprocess_multiple(img)
        
        configs = [
            # Basic configurations
            '--psm 8 --oem 3',
            '--psm 7 --oem 3', 
            '--psm 6 --oem 3',
            '--psm 13',
            
            # Alphanumeric only (common in CAPTCHAs)
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            
            # Uppercase and numbers only (common)
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            
            # Lowercase and numbers only (for cases like "bxawf8")
            '--psm 8 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789',
            '--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789',
            '--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789',
            
            # Special configurations for difficult CAPTCHAs
            '--psm 8 --oem 1',
            '--psm 7 --oem 1',
            '--psm 8 --oem 2',
            '--psm 10 --oem 3',
            
            # With additional configurations
            '--psm 8 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789 -c load_system_dawg=0 -c load_freq_dawg=0',
            '--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789 -c load_system_dawg=0 -c load_freq_dawg=0'
        ]
        
        print(f"🔍 Testing {len(processed_images)} preprocessing types x {len(configs)} OCR configurations...")
        
        for prep_name, prep_img in processed_images:
            for config in configs:
                try:
                    result = pytesseract.image_to_string(prep_img, config=config).strip()
                    # Filter results more strictly for CAPTCHAs with lines
                    if result and len(result) >= 3 and len(result) <= 12:
                        # Clean result
                        cleaned = ''.join(c for c in result if c.isalnum())
                        
                        # Additional filters to discard OCR confused by lines
                        if (len(cleaned) >= 3 and 
                            not cleaned.lower() in ['bets', 'bests', 'pets', 'pats', 'ets', 'ests', 'sees', 'sess', 'pss', 'ess'] and
                            not all(c in 'se' for c in cleaned.lower()) and  # Avoid only 's' and 'e'
                            not cleaned.lower().startswith('dav') and  # Avoid "Davessi" etc
                            len(set(cleaned.lower())) > 2):  # Must have at least 3 different chars
                            
                            results.append((f'OCR_{prep_name}', cleaned))
                except Exception:
                    continue
        
        return results
    
    def solve_with_2captcha(self, image_data):
        """Solve with 2captcha"""
        if '2captcha' not in self.services:
            return None
            
        try:
            base64_data = image_data.split(',')[1] if image_data.startswith('data:image') else image_data
            
            submit_url = "http://2captcha.com/in.php"
            data = {
                'key': self.services['2captcha'],
                'method': 'base64',
                'body': base64_data
            }
            
            response = requests.post(submit_url, data=data, timeout=30)
            if not response.text.startswith('OK|'):
                return None
                
            captcha_id = response.text.split('|')[1]
            
            for attempt in range(24):
                time.sleep(5)
                result_url = f"http://2captcha.com/res.php?key={self.services['2captcha']}&action=get&id={captcha_id}"
                result = requests.get(result_url, timeout=30)
                
                if result.text.startswith('OK|'):
                    return result.text.split('|')[1]
                elif result.text == 'CAPCHA_NOT_READY':
                    continue
                else:
                    break
                    
        except Exception as e:
            print(f"   ❌ Error 2captcha: {e}")
            
        return None
    
    def solve_with_anticaptcha(self, image_data):
        """Solve with AntiCaptcha"""
        if 'anticaptcha' not in self.services:
            return None
            
        try:
            base64_data = image_data.split(',')[1] if image_data.startswith('data:image') else image_data
            
            create_url = "https://api.anti-captcha.com/createTask"
            task_data = {
                "clientKey": self.services['anticaptcha'],
                "task": {
                    "type": "ImageToTextTask",
                    "body": base64_data
                }
            }
            
            response = requests.post(create_url, json=task_data, timeout=30)
            result = response.json()
            
            if result.get('errorId') != 0:
                return None
                
            task_id = result['taskId']
            
            for attempt in range(24):
                time.sleep(5)
                get_url = "https://api.anti-captcha.com/getTaskResult"
                get_data = {
                    "clientKey": self.services['anticaptcha'],
                    "taskId": task_id
                }
                
                response = requests.post(get_url, json=get_data, timeout=30)
                result = response.json()
                
                if result.get('status') == 'ready':
                    return result['solution']['text']
                elif result.get('status') == 'processing':
                    continue
                else:
                    break
                    
        except Exception as e:
            print(f"   ❌ Error AntiCaptcha: {e}")
            
        return None
    
    def solve_with_capmonster(self, image_data):
        """Solve with CapMonster"""
        if 'capmonster' not in self.services:
            return None
            
        try:
            base64_data = image_data.split(',')[1] if image_data.startswith('data:image') else image_data
            
            create_url = "https://api.capmonster.cloud/createTask"
            task_data = {
                "clientKey": self.services['capmonster'],
                "task": {
                    "type": "ImageToTextTask",
                    "body": base64_data
                }
            }
            
            response = requests.post(create_url, json=task_data, timeout=30)
            result = response.json()
            
            if result.get('errorId') != 0:
                return None
                
            task_id = result['taskId']
            
            for attempt in range(24):
                time.sleep(5)
                get_url = "https://api.capmonster.cloud/getTaskResult"
                get_data = {
                    "clientKey": self.services['capmonster'],
                    "taskId": task_id
                }
                
                response = requests.post(get_url, json=get_data, timeout=30)
                result = response.json()
                
                if result.get('status') == 'ready':
                    return result['solution']['text']
                elif result.get('status') == 'processing':
                    continue
                else:
                    break
                    
        except Exception as e:
            print(f"   ❌ Error CapMonster: {e}")
            
        return None
    
    def smart_consensus(self, all_results):
        """Smart consensus that weighs different sources"""
        if not all_results:
            return None, 0, []
        
        # Separate by source type
        external_services = []
        ocr_results = []
        
        for method, solution in all_results:
            if method in ['2captcha', 'AntiCaptcha', 'CapMonster']:
                external_services.append((method, solution))
            else:
                ocr_results.append((method, solution))
        
        # Frequency analysis
        all_solutions = [result[1] for result in all_results]
        frequency = Counter(all_solutions)
        
        print(f"📊 Consensus analysis:")
        for solution, count in frequency.most_common():
            sources = [result[0] for result in all_results if result[1] == solution]
            external_count = sum(1 for s in sources if s in ['2captcha', 'AntiCaptcha', 'CapMonster'])
            ocr_count = len(sources) - external_count
            
            print(f"   '{solution}': {count} times")
            if external_count > 0:
                print(f"      External services: {external_count}")
            if ocr_count > 0:
                print(f"      Local OCR: {ocr_count}")
        
        # Decision logic
        # 1. If there's consensus among external services, use it
        if len(external_services) >= 2:
            external_solutions = [result[1] for result in external_services]
            external_freq = Counter(external_solutions)
            if external_freq.most_common(1)[0][1] >= 2:
                best_solution = external_freq.most_common(1)[0][0]
                confidence = min(95, 70 + external_freq.most_common(1)[0][1] * 10)
                sources = [result[0] for result in external_services if result[1] == best_solution]
                return best_solution, confidence, sources
        
        # 2. If there's one external service, prioritize over OCR
        if external_services:
            best_solution = external_services[0][1]
            confidence = 75
            sources = [external_services[0][0]]
            
            # Check if OCR agrees
            ocr_solutions = [result[1] for result in ocr_results]
            if best_solution in ocr_solutions:
                confidence = 85
                sources.extend([result[0] for result in ocr_results if result[1] == best_solution][:2])
            
            return best_solution, confidence, sources
        
        # 3. Only OCR available - look for consensus
        if ocr_results:
            best_solution, count = frequency.most_common(1)[0]
            confidence = min(70, 30 + count * 15)
            sources = [result[0] for result in ocr_results if result[1] == best_solution][:3]
            return best_solution, confidence, sources
        
        return None, 0, []
    
    def solve_captcha(self, image_data, save_debug=False):
        """Main improved method"""
        print("\n" + "="*80)
        print("🎯 STARTING CAPTCHA SOLVING PROCESS")
        print("="*80)
        
        all_results = []
        
        # 1. Local OCR
        print("📝 Running local OCR...")
        ocr_results = self.solve_with_advanced_ocr(image_data)
        all_results.extend(ocr_results)
        
        if ocr_results:
            print(f"   ✅ OCR: {len(ocr_results)} results")
        else:
            print("   ❌ OCR no results")
        
        # 2. External services
        print("🌐 Testing external services...")
        
        if '2captcha' in self.services:
            print("   🔹 Testing 2captcha...")
            result_2captcha = self.solve_with_2captcha(image_data)
            if result_2captcha:
                all_results.append(('2captcha', result_2captcha))
                print(f"      ✅ 2captcha: '{result_2captcha}'")
            else:
                print("      ❌ 2captcha failed")
        
        if 'anticaptcha' in self.services:
            print("   🔹 Testing AntiCaptcha...")
            result_anticaptcha = self.solve_with_anticaptcha(image_data)
            if result_anticaptcha:
                all_results.append(('AntiCaptcha', result_anticaptcha))
                print(f"      ✅ AntiCaptcha: '{result_anticaptcha}'")
            else:
                print("      ❌ AntiCaptcha failed")
        
        if 'capmonster' in self.services:
            print("   🔹 Testing CapMonster...")
            result_capmonster = self.solve_with_capmonster(image_data)
            if result_capmonster:
                all_results.append(('CapMonster', result_capmonster))
                print(f"      ✅ CapMonster: '{result_capmonster}'")
            else:
                print("      ❌ CapMonster failed")
        
        # 3. Smart consensus
        print("\n🎯 Analyzing consensus...")
        best_solution, confidence, sources = self.smart_consensus(all_results)
        
        # Debug
        if save_debug:
            try:
                base64_data = image_data.split(',')[1] if image_data.startswith('data:image') else image_data
                img_bytes = base64.b64decode(base64_data)
                img = Image.open(BytesIO(img_bytes))
                img.save('debug_s0lvcaptcha_original.png')
                print("🖼️  Debug saved: debug_s0lvcaptcha_original.png")
            except Exception as e:
                print(f"Debug error: {e}")
        
        return all_results, best_solution, confidence, sources
    
    def solve_from_file(self, image_path, save_debug=False):
        """Solve from file"""
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            
            img_b64 = base64.b64encode(img_bytes).decode()
            
            if image_path.lower().endswith('.png'):
                image_data = f"data:image/png;base64,{img_b64}"
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                image_data = f"data:image/jpeg;base64,{img_b64}"
            else:
                image_data = f"data:image/png;base64,{img_b64}"
            
            print(f"📁 Processing: {image_path}")
            return self.solve_captcha(image_data, save_debug)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return [], None, 0, []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='s0lvcaptcha - Multi-service CAPTCHA solver')
    parser.add_argument('-i', '--image', help='Image file path')
    parser.add_argument('-u', '--url', help='Data:image URL')
    parser.add_argument('-c', '--config', action='store_true', help='Manage configuration')
    parser.add_argument('--reset', action='store_true', help='Reset configuration')
    
    args = parser.parse_args()
    
    # Reset command - only clears and exits
    if args.reset:
        solver = S0lvCaptcha()
        solver.reset_config()
        exit(0)
    
    # Create solver
    solver = S0lvCaptcha()
    
    # Config command
    if args.config:
        solver.manage_config()
        exit(0)
    
    # Solve CAPTCHA
    if args.image:
        if os.path.exists(args.image):
            all_results, best_solution, confidence, sources = solver.solve_from_file(args.image, save_debug=True)
        else:
            print(f"❌ File not found: {args.image}")
            all_results, best_solution, confidence, sources = [], None, 0, []
            
    elif args.url:
        if args.url.startswith('data:image'):
            all_results, best_solution, confidence, sources = solver.solve_captcha(args.url, save_debug=True)
        else:
            print("❌ Must start with 'data:image'")
            all_results, best_solution, confidence, sources = [], None, 0, []
    
    else:
        # Interactive mode
        print("\n📋 INTERACTIVE MODE")
        print("1. Solve from file")
        print("2. Solve from data:image")
        print("3. Manage APIs")
        
        choice = input("\nOption (1-3): ").strip()
        
        if choice == "1":
            image_path = input("📁 File: ").strip()
            if os.path.exists(image_path):
                all_results, best_solution, confidence, sources = solver.solve_from_file(image_path, save_debug=True)
            else:
                print(f"❌ File not found: {image_path}")
                all_results, best_solution, confidence, sources = [], None, 0, []
                
        elif choice == "2":
            data_url = input("🔗 Data:image: ").strip()
            if data_url.startswith('data:image'):
                all_results, best_solution, confidence, sources = solver.solve_captcha(data_url, save_debug=True)
            else:
                print("❌ Invalid format")
                all_results, best_solution, confidence, sources = [], None, 0, []
                
        elif choice == "3":
            solver.manage_config()
            exit(0)
            
        else:
            print("❌ Invalid option")
            all_results, best_solution, confidence, sources = [], None, 0, []
    
    # Final results
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS")
    print("="*80)
    
    if all_results:
        print(f"📊 ALL RESULTS ({len(all_results)}):")
        
        # Separate by type
        external_results = [r for r in all_results if r[0] in ['2captcha', 'AntiCaptcha', 'CapMonster']]
        ocr_results = [r for r in all_results if r[0] not in ['2captcha', 'AntiCaptcha', 'CapMonster']]
        
        if external_results:
            print("   🌐 External services:")
            for method, solution in external_results:
                print(f"      ✅ {method}: '{solution}'")
        
        if ocr_results:
            print("   🔍 Local OCR:")
            for method, solution in ocr_results[:5]:  # Show only first 5
                print(f"      ✅ {method}: '{solution}'")
            if len(ocr_results) > 5:
                print(f"      ... and {len(ocr_results) - 5} more")
    else:
        print("💀 No results")
    
    # Final recommendation
    print(f"\n💡 RECOMMENDATION: ", end="")
    if best_solution:
        print(f'"{best_solution}"')
        if confidence > 0:
            print(f"📊 Confidence: {confidence}% | Sources: {', '.join(sources[:3])}")
    else:
        print("❌ No reliable solution")
    
    print("\n🔧 COMMANDS:")
    print("   python s0lvcaptcha.py -c      # Manage APIs")
    print("   python s0lvcaptcha.py --reset # Clear configuration")
