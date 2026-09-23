FROM node:22-alpine AS build

WORKDIR /web
COPY app/web/package.json app/web/package-lock.json ./
RUN npm ci
COPY app/web/ ./

ARG VITE_TOUR=1
ENV VITE_TOUR=$VITE_TOUR
RUN npm run build

FROM nginx:1.27-alpine

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
