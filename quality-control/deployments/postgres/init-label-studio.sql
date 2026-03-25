CREATE USER label_studio WITH PASSWORD 'label_studio';
CREATE DATABASE label_studio OWNER label_studio;
GRANT ALL PRIVILEGES ON DATABASE label_studio TO label_studio;
