# Building Recognition App

A web application that identifies buildings from uploaded images and provides information from Wikipedia.

## Development

```bash
npm install --legacy-peer-deps
npm run dev
```

## Deployment to GitHub Pages

### Automatic Deployment (Recommended)
1. Push your code to the `main` branch
2. GitHub Actions will automatically build and deploy to GitHub Pages
3. Your app will be available at: https://rafzal2020.github.io/building-recognition

### Manual Deployment
```bash
npm run deploy
```

## Build
```bash
npm run build
```

## Technologies Used
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Flask (Backend)
- TensorFlow (ML Model) 