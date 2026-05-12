import "./globals.css";

export const metadata = {
  title: "ten — video search",
  description: "Open-weight video search",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
