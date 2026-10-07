FROM nginx:1.27-alpine
COPY apps/ace/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY apps/ace/frontend/index.html /usr/share/nginx/html/index.html
COPY apps/ace/frontend/app.js /usr/share/nginx/html/app.js
COPY apps/ace/frontend/styles.css /usr/share/nginx/html/styles.css
