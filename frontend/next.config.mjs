/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    RENDER_API_URL: process.env.RENDER_API_URL,
  },

}

export default nextConfig
