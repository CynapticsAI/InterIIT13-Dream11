# Node version
FROM node:18

WORKDIR /app

COPY package*.json ./
RUN npm install --force

COPY . .

EXPOSE 3000

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
