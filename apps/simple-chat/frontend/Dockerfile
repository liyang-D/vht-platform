FROM node:24-alpine AS build

WORKDIR /app

COPY apps/simple-chat/frontend/package.json apps/simple-chat/frontend/package-lock.json ./
RUN npm ci

COPY apps/simple-chat/frontend ./
RUN npm run build

FROM nginx:1.27-alpine

ENV BACKEND_UPSTREAM=http://simple-chat-backend:8010

COPY apps/simple-chat/frontend/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
