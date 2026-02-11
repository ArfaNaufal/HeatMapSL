# Stage 1: Build
FROM node:20-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
# Stage 2: Serve
FROM nginx:stable-alpine
# Copy your build folder
COPY --from=build /app/dist /usr/share/nginx/html
# Copy your custom config
COPY nginx.conf /etc/nginx/conf.d/default.conf
# You might need a custom nginx.conf to handle React Router paths
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]