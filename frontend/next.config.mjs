// next.config.mjs
import dotenv from "dotenv";
dotenv.config();

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_RENDER_API_URL: process.env.NEXT_PUBLIC_RENDER_API_URL,
  },
};

export default nextConfig;
