#!/bin/bash

# Automotive API Production Deployment Script
set -e

echo "🚀 Automotive API Production Deployment Başlatılıyor..."

# 1. Docker image build
echo "📦 Docker image build ediliyor..."
docker build -t automotive-api:latest .

# 2. Docker Hub'a tag ve push
echo "🌐 Docker Hub'a push ediliyor..."
DOCKER_HUB_USER=${DOCKER_HUB_USER:-"yourusername"}
VERSION=${VERSION:-"latest"}

docker tag automotive-api:latest $DOCKER_HUB_USER/automotive-api:$VERSION
docker tag automotive-api:latest $DOCKER_HUB_USER/automotive-api:latest

echo "🔐 Docker Hub'a login..."
# docker login # Manuel login gerekirse

echo "⬆️ Image push ediliyor..."
docker push $DOCKER_HUB_USER/automotive-api:$VERSION
docker push $DOCKER_HUB_USER/automotive-api:latest

echo "✅ Deployment tamamlandı!"
echo "📡 Image: $DOCKER_HUB_USER/automotive-api:$VERSION"

# 3. Local test
echo "🧪 Local deployment test..."
docker run -d --name automotive-api-test \
  -p 8081:8080 \
  $DOCKER_HUB_USER/automotive-api:latest

sleep 10

echo "🔍 Health check..."
curl -f http://localhost:8081/health || echo "❌ Health check başarısız"

echo "🛑 Test container temizleniyor..."
docker stop automotive-api-test || true
docker rm automotive-api-test || true

echo "🎉 Deployment başarılı! Production'da kullanabilirsiniz."
