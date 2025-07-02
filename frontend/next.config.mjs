/** @type {import('next').NextConfig} */
const isGithubPages = process.env.GITHUB_PAGES === 'true'
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  assetPrefix: isGithubPages ? '/building-recognition/' : '',
  basePath: isGithubPages ? '/<building-recognition>' : '',
  trailingSlash: true,
  output: 'export', // static export for GitHub Pages
}

export default nextConfig
