#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import shutil
import json

class CreacionDatasetReto4:
    def __init__(self):
        # Detectar si estamos en Punto4 o en la raiz
        current_dir = os.getcwd()
        if current_dir.endswith('Punto4'):
            self.base_path = "."
            self.punto1_path = os.path.join("..", "Punto1")
        else:
            # Estamos en la raiz ParcialIA
            self.base_path = "Punto4"
            self.punto1_path = "Punto1"
        
        self.region_name = "Santa_Marta_Colombia"
        self.patch_size = 512
        self.overlap_ratio = 0.5
        self.ensure_folder_structure()
        
    def ensure_folder_structure(self):
        """Crear estructura organizada para Reto 4"""
        folders = [
            "input_images",        
            "registered_images",   
            "fusion_results",      
            "dataset_patches",     
            "dataset_patches/noisy",      
            "dataset_patches/ground_truth", 
            "validation",          
            "metrics"             
        ]
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        
        for folder in folders:
            folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        print("Estructura de carpetas creada en {}".format(self.base_path))
        
    def load_scaled_images_from_punto1(self):
        """Cargar imagenes escaladas del Punto1 y normalizarlas"""
        print("\nPASO 1: Cargando imagenes escaladas del Punto1...")
        
        scaled_path = os.path.join(self.punto1_path, "scaled_images")
        
        if not os.path.exists(scaled_path):
            print("  ERROR: No existe carpeta scaled_images en {}".format(scaled_path))
            return []
        
        images = []
        image_files = []
        
        try:
            files = os.listdir(scaled_path)
            png_files = [f for f in files if f.lower().endswith('.png')]
            png_files.sort()
            
            print("  Archivos encontrados: {}".format(png_files))
            
            # Primero cargar todas las imágenes para encontrar el tamaño mínimo común
            temp_images = []
            shapes = []
            
            for filename in png_files[:5]:
                file_path = os.path.join(scaled_path, filename)
                image = np.array(Image.open(file_path))
                
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2).astype(np.uint8)
                
                temp_images.append(image)
                shapes.append(image.shape)
                print("  Imagen cargada: {} -> {}".format(image.shape, filename))
            
            # Encontrar tamaño mínimo común
            min_height = min([s[0] for s in shapes])
            min_width = min([s[1] for s in shapes])
            target_shape = (min_height, min_width)
            
            print("  Tamaño objetivo común: {}".format(target_shape))
            
            # Redimensionar todas las imágenes al tamaño común
            for i, (image, filename) in enumerate(zip(temp_images, png_files[:5])):
                # Recortar al tamaño común (desde el centro)
                h, w = image.shape
                start_h = (h - min_height) // 2
                start_w = (w - min_width) // 2
                
                cropped_image = image[start_h:start_h + min_height, start_w:start_w + min_width]
                
                images.append(cropped_image)
                image_files.append(filename)
                
                # Guardar imagen normalizada
                dest_path = os.path.join(self.base_path, "input_images", "image_{}.png".format(i+1))
                Image.fromarray(cropped_image).save(dest_path)
                
                print("  Imagen {} normalizada: {} -> {}".format(i+1, image.shape, cropped_image.shape))
            
            print("  {} imagenes normalizadas exitosamente".format(len(images)))
            return images, image_files
            
        except Exception as e:
            print("  ERROR cargando imagenes: {}".format(e))
            return [], []
    
    def select_base_image(self, images):
        """Seleccionar imagen base para registro"""
        print("\nPASO 2: Seleccionando imagen base...")
        
        if len(images) < 3:
            print("  ERROR: Se necesitan al menos 3 imagenes")
            return None, -1
        
        base_index = len(images) // 2
        base_image = images[base_index]
        
        print("  Imagen base seleccionada: Imagen {} de {}".format(base_index + 1, len(images)))
        print("  Dimensiones imagen base: {}".format(base_image.shape))
        
        base_path = os.path.join(self.base_path, "input_images", "base_image.png")
        Image.fromarray(base_image).save(base_path)
        
        return base_image, base_index
    
    def register_images_simple(self, base_image, images, base_index):
        """Registro simple sin OpenCV (usando correlación NumPy)"""
        print("\nPASO 3: Registrando imagenes a la base...")
        
        registered_images = []
        registration_info = []
        
        for i, image in enumerate(images):
            if i == base_index:
                registered_images.append(image.copy())
                registration_info.append({"image": i+1, "shift_x": 0, "shift_y": 0, "method": "base"})
                print("  Imagen {}: Base de referencia".format(i+1))
            else:
                print("  Registrando imagen {} a la base...".format(i+1))
                
                try:
                    # Registro simple usando correlación NumPy
                    registered_img, shift_x, shift_y = self.numpy_correlation_registration(base_image, image)
                    registered_images.append(registered_img)
                    registration_info.append({
                        "image": i+1, 
                        "shift_x": int(shift_x), 
                        "shift_y": int(shift_y), 
                        "method": "numpy_correlation"
                    })
                    print("    Desplazamiento: x={}, y={}".format(shift_x, shift_y))
                    
                    reg_path = os.path.join(self.base_path, "registered_images", "registered_{}.png".format(i+1))
                    Image.fromarray(registered_img).save(reg_path)
                    
                except Exception as e:
                    print("    ERROR en registro: {}".format(e))
                    registered_images.append(image.copy())
                    registration_info.append({"image": i+1, "shift_x": 0, "shift_y": 0, "method": "failed"})
        
        return registered_images, registration_info
    
    def numpy_correlation_registration(self, base_image, target_image):
        """Registro usando correlación con NumPy puro"""
        # Asegurar que ambas imágenes tengan el mismo tamaño
        if base_image.shape != target_image.shape:
            print("    Imagenes con diferentes tamaños, usando imagen original")
            return target_image.copy(), 0, 0
        
        # Normalizar intensidades
        base_norm = (base_image.astype(np.float32) - np.mean(base_image)) / (np.std(base_image) + 1e-8)
        target_norm = (target_image.astype(np.float32) - np.mean(target_image)) / (np.std(target_image) + 1e-8)
        
        # Buscar desplazamiento óptimo en una ventana pequeña
        max_shift = min(50, min(base_image.shape) // 10)  # Máximo 50 píxeles o 10% de la imagen
        best_correlation = -np.inf
        best_shift = (0, 0)
        
        for dy in range(-max_shift, max_shift + 1, 5):  # Paso de 5 para eficiencia
            for dx in range(-max_shift, max_shift + 1, 5):
                # Calcular región de overlap
                y1, y2 = max(0, dy), min(base_image.shape[0], base_image.shape[0] + dy)
                x1, x2 = max(0, dx), min(base_image.shape[1], base_image.shape[1] + dx)
                
                ty1, ty2 = max(0, -dy), min(target_image.shape[0], target_image.shape[0] - dy)
                tx1, tx2 = max(0, -dx), min(target_image.shape[1], target_image.shape[1] - dx)
                
                if (y2 - y1) > 100 and (x2 - x1) > 100:  # Región mínima de overlap
                    base_region = base_norm[y1:y2, x1:x2]
                    target_region = target_norm[ty1:ty2, tx1:tx2]
                    
                    # Correlación normalizada
                    correlation = np.corrcoef(base_region.flatten(), target_region.flatten())[0, 1]
                    
                    if not np.isnan(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                        best_shift = (dx, dy)
        
        dx, dy = best_shift
        
        # Aplicar desplazamiento
        registered = np.zeros_like(target_image)
        
        # Calcular regiones válidas
        y1, y2 = max(0, dy), min(base_image.shape[0], target_image.shape[0] + dy)
        x1, x2 = max(0, dx), min(base_image.shape[1], target_image.shape[1] + dx)
        
        ty1, ty2 = max(0, -dy), min(target_image.shape[0], target_image.shape[0] - dy)
        tx1, tx2 = max(0, -dx), min(target_image.shape[1], target_image.shape[1] - dx)
        
        # Copiar región válida
        if y2 > y1 and x2 > x1:
            registered[y1:y2, x1:x2] = target_image[ty1:ty2, tx1:tx2]
        else:
            # Si no se puede registrar, usar imagen original
            registered = target_image.copy()
            dx, dy = 0, 0
        
        return registered, dx, dy
    
    def create_fusion_image(self, registered_images):
        """Crear imagen fusionada usando media temporal"""
        print("\nPASO 4: Creando imagen fusionada (ground truth)...")
        
        if len(registered_images) == 0:
            return None
        
        # Verificar que todas las imágenes tengan el mismo tamaño
        base_shape = registered_images[0].shape
        valid_images = []
        
        for i, img in enumerate(registered_images):
            if img.shape == base_shape:
                valid_images.append(img)
                print("  Imagen {} incluida en fusion: {}".format(i+1, img.shape))
            else:
                print("  Imagen {} excluida por tamaño diferente: {}".format(i+1, img.shape))
        
        if len(valid_images) == 0:
            print("  ERROR: No hay imagenes validas para fusion")
            return None
        
        # Convertir a array 3D y calcular media
        image_stack = np.stack(valid_images, axis=0).astype(np.float32)
        fusion_image = np.mean(image_stack, axis=0).astype(np.uint8)
        
        print("  Fusion completada: {} imagenes promediadas".format(len(valid_images)))
        print("  Imagen fusionada: {}".format(fusion_image.shape))
        
        fusion_path = os.path.join(self.base_path, "fusion_results", "fusion_ground_truth.png")
        Image.fromarray(fusion_image).save(fusion_path)
        
        return fusion_image
    
    def generate_patches_512x512(self, base_image, fusion_image):
        """Generar patches de 512x512 con overlap"""
        print("\nPASO 5: Generando patches de {}x{}...".format(self.patch_size, self.patch_size))
        
        if base_image is None or fusion_image is None:
            print("  ERROR: Imagenes base o fusionada son None")
            return 0
        
        # Verificar que ambas imágenes tengan el mismo tamaño
        if base_image.shape != fusion_image.shape:
            print("  ERROR: Imagenes base y fusionada tienen tamaños diferentes")
            print("  Base: {}, Fusion: {}".format(base_image.shape, fusion_image.shape))
            return 0
        
        step = int(self.patch_size * (1 - self.overlap_ratio))
        patch_count = 0
        
        h, w = base_image.shape
        print("  Imagen completa: {}x{}".format(h, w))
        print("  Tamaño patch: {}x{}, Paso: {}, Overlap: {}%".format(
            self.patch_size, self.patch_size, step, self.overlap_ratio * 100))
        
        # Verificar que la imagen sea suficientemente grande
        if h < self.patch_size or w < self.patch_size:
            print("  ERROR: Imagen demasiado pequeña para patches de {}x{}".format(self.patch_size, self.patch_size))
            print("  Tamaño mínimo requerido: {}x{}".format(self.patch_size, self.patch_size))
            return 0
        
        # Generar patches
        for y in range(0, h - self.patch_size + 1, step):
            for x in range(0, w - self.patch_size + 1, step):
                base_patch = base_image[y:y+self.patch_size, x:x+self.patch_size]
                fusion_patch = fusion_image[y:y+self.patch_size, x:x+self.patch_size]
                
                if self.validate_patch(base_patch, fusion_patch):
                    patch_count += 1
                    
                    base_filename = "patch_{:03d}.png".format(patch_count)
                    fusion_filename = "patch_{:03d}.png".format(patch_count)
                    
                    base_patch_path = os.path.join(self.base_path, "dataset_patches", "noisy", base_filename)
                    fusion_patch_path = os.path.join(self.base_path, "dataset_patches", "ground_truth", fusion_filename)
                    
                    Image.fromarray(base_patch).save(base_patch_path)
                    Image.fromarray(fusion_patch).save(fusion_patch_path)
        
        print("  {} patches validos generados".format(patch_count))
        return patch_count
    
    def validate_patch(self, base_patch, fusion_patch):
        """Validar que el patch tenga contenido util"""
        base_std = np.std(base_patch)
        fusion_std = np.std(fusion_patch)
        
        if base_std < 5 or fusion_std < 5:
            return False
        
        extreme_ratio = (np.sum(base_patch < 10) + np.sum(base_patch > 245)) / base_patch.size
        if extreme_ratio > 0.3:
            return False
        
        return True
    
    def create_dataset_metadata(self, registration_info, patch_count, base_index):
        """Crear metadata del dataset"""
        print("\nPASO 6: Creando metadata del dataset...")
        
        metadata = {
            "dataset_info": {
                "name": "SAR_Denoising_Dataset_{}".format(self.region_name),
                "description": "Dataset para denoising de imagenes SAR usando fusion temporal",
                "region": self.region_name,
                "creation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "patch_size": [self.patch_size, self.patch_size],
                "total_patches": patch_count,
                "overlap_ratio": self.overlap_ratio
            },
            "processing_info": {
                "base_image_index": base_index + 1,
                "fusion_method": "temporal_averaging",
                "registration_method": "numpy_correlation",
                "images_used": len(registration_info)
            },
            "registration_details": registration_info,
            "dataset_structure": {
                "noisy": "Patches de imagen base con speckle natural",
                "ground_truth": "Patches de imagen fusionada (denoised)"
            }
        }
        
        metadata_path = os.path.join(self.base_path, "dataset_info.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        readme_content = [
            "DATASET SAR DENOISING - {}".format(self.region_name),
            "="*50,
            "Creado: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')),
            "",
            "ESTRUCTURA:",
            "dataset_patches/",
            "├── noisy/          # Imagen ruidosa (input)",
            "└── ground_truth/   # Imagen limpia (target)",
            "",
            "ESTADISTICAS:",
            "- Total de patches: {}".format(patch_count),
            "- Tamaño patches: {}x{} pixeles".format(self.patch_size, self.patch_size),
            "- Overlap: {}%".format(self.overlap_ratio * 100)
        ]
        
        readme_path = os.path.join(self.base_path, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(readme_content))
        
        print("  Metadata guardada: dataset_info.json")
        print("  README creado: README.txt")
    
    def generate_final_report(self, patch_count, registration_info):
        """Generar reporte final"""
        print("\nPASO 7: Generando reporte final...")
        
        report = []
        report.append("="*80)
        report.append("RETO 4: CREACION DEL DATASET - REPORTE COMPLETO")
        report.append("="*80)
        report.append("Region: {}".format(self.region_name))
        report.append("Fecha: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        report.append("")
        report.append("RESUMEN:")
        report.append("• Dataset SAR denoising creado exitosamente")
        report.append("• {} patches de {}x{} pixeles".format(patch_count, self.patch_size, self.patch_size))
        report.append("• {} imagenes procesadas".format(len(registration_info)))
        report.append("")
        report.append("RETO 4 COMPLETADO - VALE 50% DEL PARCIAL")
        report.append("="*80)
        
        report_path = os.path.join(self.base_path, "metrics", "reporte_reto4_completo.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("  Reporte guardado: reporte_reto4_completo.txt")
        
        print("\n" + "="*60)
        print("RESUMEN FINAL - RETO 4")
        print("="*60)
        print("Dataset: {} patches de {}x{}".format(patch_count, self.patch_size, self.patch_size))
        print("COMPLETADO - Vale 50% del parcial")
    
    def execute_complete_pipeline(self):
        """Ejecutar pipeline completo"""
        print("INICIANDO RETO 4: CREACION DEL DATASET")
        print("="*80)
        
        try:
            images, image_files = self.load_scaled_images_from_punto1()
            if len(images) == 0:
                return False
            
            base_image, base_index = self.select_base_image(images)
            if base_image is None:
                return False
            
            registered_images, registration_info = self.register_images_simple(base_image, images, base_index)
            
            fusion_image = self.create_fusion_image(registered_images)
            if fusion_image is None:
                return False
            
            patch_count = self.generate_patches_512x512(base_image, fusion_image)
            if patch_count == 0:
                return False
            
            self.create_dataset_metadata(registration_info, patch_count, base_index)
            self.generate_final_report(patch_count, registration_info)
            
            print("\nRETO 4 COMPLETADO EXITOSAMENTE!")
            return True
            
        except Exception as e:
            print("ERROR: {}".format(e))
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("="*80)
    print("RETO 4: CREACION DEL DATASET")
    print("Valor: 50% del parcial")
    print("="*80)
    
    dataset_creator = CreacionDatasetReto4()
    resultado = dataset_creator.execute_complete_pipeline()
    
    if resultado:
        print("\nDataset creado exitosamente")
    else:
        print("\nError en la creacion del dataset")