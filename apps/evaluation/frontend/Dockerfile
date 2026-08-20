FROM node:24-alpine AS build

WORKDIR /app

COPY apps/evaluation/frontend/package.json apps/evaluation/frontend/package-lock.json ./
RUN npm ci

COPY apps/evaluation/frontend ./
ARG VITE_APP_BASE_PATH=/evaluation/
ENV VITE_APP_BASE_PATH=${VITE_APP_BASE_PATH}
RUN npm run build

FROM nginx:1.27-alpine

ENV BACKEND_UPSTREAM=http://evaluation-backend:8050 \
    BACKEND_PROXY_TIMEOUT=600s

COPY apps/evaluation/frontend/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
