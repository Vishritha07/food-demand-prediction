# Platewise – Food Demand & Wastage Prediction

A React + Vite frontend prototype for a multi-canteen food demand and wastage prediction platform.

## Current frontend flow

1. Landing page
2. Animated `Get started` loading button (0–100%)
3. Login / Create account choice
4. Login or canteen registration form
5. Empty canteen administrator dashboard

## Dashboard sections

- Overview
- Food items
- Historical data upload and preview (CSV / XLSX / XLS)
- Predict demand (UI only; prediction remains disabled until the trained model/API is connected)
- Actual records
- Reports and analytics
- Settings

## Run locally

Use Node.js (recommended: current LTS) and npm.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173/`.

## Important prototype notes

- The application starts with **no preloaded food items, canteen profile, records, chart values, or prediction results**.
- Login and registration are frontend previews only. They do not create accounts, authenticate users, or store passwords.
- Menu items and actual records entered in the dashboard remain in React memory for the current browser session only. Refreshing or signing out clears them.
- Uploading a CSV or Excel file reads it for a temporary preview only. It is not imported into a database.
- Demand prediction is intentionally disabled until the Random Forest model and Flask API are integrated. No mock prediction is generated.
- Flask, MySQL, real authentication, permanent data storage, and ML integration are not included in this frontend-only version.

## Technologies

- React 18
- Vite 6
- CSS
- Recharts
- PapaParse
- SheetJS (`xlsx`)
- Lucide React
