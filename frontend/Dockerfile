# frontend/Dockerfile (multi-stage)
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2 - serve with nginx
FROM nginx:1.25-alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# optional: custom nginx config if you want API proxying; otherwise default config serves static files
# Copy custom conf if present
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
