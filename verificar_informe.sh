#!/bin/bash
# Script para verificar que tienes todos los archivos necesarios para Overleaf

echo "🔍 Verificando archivos para el informe LaTeX..."
echo ""

ARCHIVOS_OK=0
ARCHIVOS_FALTANTES=0

# Verificar archivo principal
if [ -f "INFORME_PIPELINE_90_CONCISO.tex" ]; then
    echo "✅ INFORME_PIPELINE_90_CONCISO.tex"
    ARCHIVOS_OK=$((ARCHIVOS_OK + 1))
else
    echo "❌ INFORME_PIPELINE_90_CONCISO.tex - FALTA"
    ARCHIVOS_FALTANTES=$((ARCHIVOS_FALTANTES + 1))
fi

# Verificar logo UNSA
if [ -f "logo_unsa.png" ]; then
    echo "✅ logo_unsa.png"
    ARCHIVOS_OK=$((ARCHIVOS_OK + 1))
else
    echo "⚠️  logo_unsa.png - FALTA (descárgalo de Google Images o UNSA website)"
    ARCHIVOS_FALTANTES=$((ARCHIVOS_FALTANTES + 1))
fi

# Verificar imagen de métricas
if [ -f "metricas_bajas.png" ]; then
    echo "✅ metricas_bajas.png"
    ARCHIVOS_OK=$((ARCHIVOS_OK + 1))
else
    echo "⚠️  metricas_bajas.png - FALTA (toma captura de pantalla de http://localhost:8000)"
    ARCHIVOS_FALTANTES=$((ARCHIVOS_FALTANTES + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Resumen:"
echo "  ✅ Archivos listos: $ARCHIVOS_OK"
echo "  ⚠️  Archivos faltantes: $ARCHIVOS_FALTANTES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $ARCHIVOS_FALTANTES -eq 0 ]; then
    echo ""
    echo "🎉 ¡Todo listo! Puedes subir a Overleaf:"
    echo ""
    echo "1. Ve a https://www.overleaf.com"
    echo "2. New Project → Upload Project"
    echo "3. Sube estos archivos:"
    echo "   - INFORME_PIPELINE_90_CONCISO.tex"
    echo "   - logo_unsa.png"
    echo "   - metricas_bajas.png"
    echo ""
    echo "4. Compila con pdfLaTeX"
    echo ""
    echo "📖 Consulta INSTRUCCIONES_OVERLEAF.md para más detalles"
else
    echo ""
    echo "📋 Archivos faltantes:"
    echo ""
    if [ ! -f "logo_unsa.png" ]; then
        echo "1. Logo UNSA:"
        echo "   - Busca en Google: 'logo unsa png'"
        echo "   - Descarga y guarda como: logo_unsa.png"
        echo ""
    fi
    if [ ! -f "metricas_bajas.png" ]; then
        echo "2. Imagen de métricas:"
        echo "   - Abre http://localhost:8000 en tu navegador"
        echo "   - Sube un PDF de prueba"
        echo "   - Toma captura de pantalla de las métricas"
        echo "   - Guarda como: metricas_bajas.png"
        echo ""
    fi
fi

echo ""
echo "💡 Tip: Puedes comentar las líneas de imágenes en el .tex si no las tienes aún"
echo "    Busca las líneas con \\includegraphics y añade % al inicio"
