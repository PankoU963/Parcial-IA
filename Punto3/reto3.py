#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import shutil

class ClasificacionAguaReto3:
    def __init__(self):
        # Detectar si estamos en Punto3 o en la raiz
        current_dir = os.getcwd()
        if current_dir.endswith('Punto3'):
            self.base_path = "."
            self.input_path = os.path.join("..", "Punto2")
        else:
            # Estamos en la raiz ParcialIA
            self.base_path = "Punto3"
            self.input_path = "Punto2"
        
        self.region_name = "PrinceTown"
        self.ensure_folder_structure()
        
    def ensure_folder_structure(self):
        """Crear estructura organizada para Reto 3"""
        folders = [
            "input_clustering",
            "water_masks",
            "analysis",
            "comparison",
            "metrics"
        ]
        
        # Crear carpeta base si no existe
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        
        for folder in folders:
            folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        print("Estructura de carpetas creada en {}".format(self.base_path))
        
    def copy_clustering_inputs(self):
        """Copiar imagenes clasificadas del Punto 2"""
        print("\nPASO 1: Copiando imagenes clasificadas del Punto 2...")
        
        # Archivos específicos encontrados en el diagnóstico
        files_to_copy = [
            ("clustered_images/filtrada_clusters.png", "filtrada_clusters.png"),
            ("clustered_images/original_clusters.png", "original_clusters.png")
        ]
        
        input_clustering_path = os.path.join(self.base_path, "input_clustering")
        files_copied = 0
        
        for source_file, dest_file in files_to_copy:
            source_path = os.path.join(self.input_path, source_file)
            dest_path = os.path.join(input_clustering_path, dest_file)
            
            print("  Buscando: {}".format(source_path))
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                size = os.path.getsize(source_path)
                print("  COPIADO: {} ({} bytes)".format(dest_file, size))
                files_copied += 1
            else:
                print("  NO ENCONTRADO: {}".format(source_path))
        
        return files_copied > 0
        
    def load_clustered_images(self):
        """Cargar las imagenes clasificadas"""
        print("\nPASO 2: Cargando imagenes clasificadas...")
        
        filtrada_path = os.path.join(self.base_path, "input_clustering", "filtrada_clusters.png")
        original_path = os.path.join(self.base_path, "input_clustering", "original_clusters.png")
        
        images = {}
        
        try:
            if os.path.exists(filtrada_path):
                images['filtrada'] = np.array(Image.open(filtrada_path))
                print("  Imagen filtrada cargada: {}".format(images['filtrada'].shape))
            else:
                print("  No encontrada: {}".format(filtrada_path))
                
            if os.path.exists(original_path):
                images['original'] = np.array(Image.open(original_path))
                print("  Imagen original cargada: {}".format(images['original'].shape))
            else:
                print("  No encontrada: {}".format(original_path))
                
        except Exception as e:
            print("  Error cargando imagenes: {}".format(e))
            
        return images
        
    def analyze_cluster_values(self, images):
        """Analizar valores unicos en las imagenes clasificadas"""
        print("\nPASO 3: Analizando valores de clustering...")
        
        analysis = {}
        
        for name, image in images.items():
            if len(image.shape) == 3:
                image = np.mean(image, axis=2).astype(np.uint8)
            
            unique_values = np.unique(image)
            pixel_counts = {}
            
            for value in unique_values:
                count = np.sum(image == value)
                percentage = (count / float(image.size)) * 100
                pixel_counts[int(value)] = {
                    'count': int(count),
                    'percentage': round(percentage, 2)
                }
            
            analysis[name] = {
                'shape': image.shape,
                'unique_values': unique_values.tolist(),
                'pixel_distribution': pixel_counts,
                'processed_image': image
            }
            
            print("\n  Analisis imagen {}:".format(name))
            print("     Dimensiones: {}".format(image.shape))
            print("     Valores unicos: {}".format(unique_values.tolist()))
            for value, stats in pixel_counts.items():
                print("     Valor {}: {} pixeles ({}%)".format(value, stats['count'], stats['percentage']))
        
        return analysis
        
    def create_water_masks(self, analysis):
        """Crear mascaras binarias agua/no-agua"""
        print("\nPASO 4: Creando mascaras binarias agua/no-agua...")
        
        water_masks = {}
        water_stats = {}
        
        for name, data in analysis.items():
            image = data['processed_image']
            unique_values = sorted(data['unique_values'])
            water_value = unique_values[0]  # Valor mas bajo = agua
            
            print("  Imagen {}: Identificando valor {} como AGUA".format(name, water_value))
            
            water_mask = np.zeros_like(image, dtype=np.uint8)
            water_mask[image == water_value] = 255
            
            water_masks[name] = water_mask
            
            water_pixels = np.sum(water_mask == 255)
            total_pixels = water_mask.size
            water_percentage = (water_pixels / float(total_pixels)) * 100
            
            water_stats[name] = {
                'water_pixels': int(water_pixels),
                'total_pixels': int(total_pixels),
                'water_percentage': round(water_percentage, 3),
                'no_water_percentage': round(100 - water_percentage, 3),
                'water_value_original': int(water_value)
            }
            
            print("     Pixeles de agua: {:,} ({:.3f}%)".format(water_pixels, water_percentage))
            
            mask_filename = "mask_agua_{}.png".format(name)
            mask_path = os.path.join(self.base_path, "water_masks", mask_filename)
            Image.fromarray(water_mask).save(mask_path)
            print("     Mascara guardada: {}".format(mask_filename))
        
        return water_masks, water_stats
        
    def create_visualizations(self, analysis, water_masks, water_stats):
        """Crear visualizaciones comparativas"""
        print("\nPASO 5: Creando visualizaciones...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('RETO 3: Clasificacion Agua/No-Agua - {}'.format(self.region_name), fontsize=16, fontweight='bold')
        
        names = ['filtrada', 'original']
        
        for i, name in enumerate(names):
            if name in analysis and name in water_masks:
                axes[i, 0].imshow(analysis[name]['processed_image'], cmap='gray')
                axes[i, 0].set_title('Clustering {}'.format(name.title()))
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(water_masks[name], cmap='gray')
                axes[i, 1].set_title('Mascara Agua {}\n{:.3f}% agua'.format(name.title(), water_stats[name]["water_percentage"]))
                axes[i, 1].axis('off')
                
                overlay = np.stack([
                    analysis[name]['processed_image'],
                    analysis[name]['processed_image'] + water_masks[name] // 4,
                    analysis[name]['processed_image']
                ], axis=2)
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title('Superposicion {}'.format(name.title()))
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(self.base_path, "analysis", "comparacion_completa.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Visualizacion guardada")
        
        if 'filtrada' in water_stats and 'original' in water_stats:
            self.create_difference_analysis(water_stats)
        
    def create_difference_analysis(self, water_stats):
        """Crear analisis de diferencias"""
        print("\nPASO 6: Analizando diferencias...")
        
        filtrada_water = water_stats['filtrada']['water_percentage']
        original_water = water_stats['original']['water_percentage']
        difference = abs(filtrada_water - original_water)
        
        # Crear grafico de barras compatible con Python 3.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Grafico de barras - usar indices numericos en lugar de strings
        water_percentages = [filtrada_water, original_water]
        x_positions = [0, 1]  # Posiciones numericas
        
        bars = ax1.bar(x_positions, water_percentages, color=['#2E86C1', '#E74C3C'])
        ax1.set_ylabel('Porcentaje de Agua (%)')
        ax1.set_title('Comparacion Deteccion de Agua')
        ax1.set_ylim(0, max(water_percentages) * 1.2)
        
        # Configurar etiquetas del eje x
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(['Imagen Filtrada', 'Imagen Original'])
        
        # Anadir valores en las barras
        for i, (bar, percentage) in enumerate(zip(bars, water_percentages)):
            height = bar.get_height()
            ax1.text(x_positions[i], height + height*0.01,
                    '{:.3f}%'.format(percentage), ha='center', va='bottom', fontweight='bold')
        
        # Grafico de diferencia
        ax2.bar([0], [difference], color='#F39C12')
        ax2.set_ylabel('Diferencia Absoluta (%)')
        ax2.set_title('Diferencia entre Metodos')
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Diferencia'])
        ax2.text(0, difference + difference*0.1, '{:.3f}%'.format(difference), 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        diff_path = os.path.join(self.base_path, "comparison", "diferencias_agua.png")
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Analisis de diferencias guardado")

        
    def generate_report(self, analysis, water_stats):
        """Generar reporte completo"""
        print("\nPASO 7: Generando reporte completo...")
        
        report = []
        report.append("="*80)
        report.append("RETO 3: CLASIFICACION AGUA/NO-AGUA - REPORTE COMPLETO")
        report.append("="*80)
        report.append("Region: {}".format(self.region_name))
        report.append("Fecha: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        report.append("")
        
        if 'filtrada' in water_stats and 'original' in water_stats:
            filtrada_water = water_stats['filtrada']['water_percentage']
            original_water = water_stats['original']['water_percentage']
            difference = abs(filtrada_water - original_water)
            
            report.append("RESUMEN EJECUTIVO:")
            report.append("• Agua detectada (filtrada): {:.3f}%".format(filtrada_water))
            report.append("• Agua detectada (original): {:.3f}%".format(original_water))
            report.append("• Diferencia: {:.3f}%".format(difference))
            report.append("")
            
            for name, stats in water_stats.items():
                report.append("IMAGEN {}:".format(name.upper()))
                report.append("  Pixeles agua: {:,}".format(stats['water_pixels']))
                report.append("  Pixeles total: {:,}".format(stats['total_pixels']))
                report.append("  Porcentaje agua: {:.3f}%".format(stats['water_percentage']))
                report.append("")
        
        report.append("RETO 3 COMPLETADO EXITOSAMENTE")
        report.append("="*80)
        
        report_path = os.path.join(self.base_path, "metrics", "reporte_reto3_completo.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("  Reporte guardado")
        
        print("\nRESUMEN FINAL:")
        if 'filtrada' in water_stats and 'original' in water_stats:
            print("Agua detectada (filtrada): {:.3f}%".format(water_stats['filtrada']['water_percentage']))
            print("Agua detectada (original): {:.3f}%".format(water_stats['original']['water_percentage']))
            difference = abs(water_stats['filtrada']['water_percentage'] - water_stats['original']['water_percentage'])
            print("Diferencia: {:.3f}%".format(difference))
            print("COMPLETADO - Vale 15% del parcial")
        
    def execute_complete_analysis(self):
        """Ejecutar analisis completo"""
        print("INICIANDO RETO 3: CLASIFICACION AGUA/NO-AGUA")
        print("="*80)
        
        try:
            if not self.copy_clustering_inputs():
                print("ERROR: No se pudieron copiar archivos del Punto2")
                return False
            
            images = self.load_clustered_images()
            if not images:
                print("ERROR: No se pudieron cargar las imagenes")
                return False
            
            analysis = self.analyze_cluster_values(images)
            water_masks, water_stats = self.create_water_masks(analysis)
            self.create_visualizations(analysis, water_masks, water_stats)
            self.generate_report(analysis, water_stats)
            
            print("\nRETO 3 COMPLETADO EXITOSAMENTE!")
            return True
            
        except Exception as e:
            print("ERROR: {}".format(e))
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("RETO 3: CLASIFICACION AGUA/NO-AGUA")
    
    clasificador = ClasificacionAguaReto3()
    resultado = clasificador.execute_complete_analysis()
    
    if resultado:
        print("\nTodos los archivos generados correctamente")
    else:
        print("\nError en el procesamiento")